import os
import math
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
import matplotlib.pyplot as plt

from dataset_generation import create_dataset_file


class ExponentialLeakyIntegrator(nn.Module):
    """Discrete-time leaky integrator."""

    def __init__(self, tau_ms: float, delta_t_ms: float, learnable_alpha: bool = False):
        super().__init__()
        alpha_val = math.exp(-delta_t_ms / tau_ms)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_val, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_val, dtype=torch.float32))

    def forward(self, x_t: torch.Tensor, prev_state: torch.Tensor | None) -> torch.Tensor:
        if prev_state is None:
            prev_state = torch.zeros_like(x_t)
        return x_t + self.alpha * prev_state


class LIConv2d(nn.Module):
    """Conv-BN-ReLU block wrapped with a leaky integrator."""

    def __init__(self, conv_block: nn.Module, tau_ms: float, delta_t_ms: float):
        super().__init__()
        self.conv_block = conv_block
        self.integrator = ExponentialLeakyIntegrator(tau_ms, delta_t_ms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        outputs = []
        prev_state = None
        for t in range(T):
            y_t = self.conv_block(x[:, t])
            v_t = self.integrator(y_t, prev_state)
            outputs.append(v_t)
            prev_state = v_t
        return torch.stack(outputs, dim=1)


class LICNN(nn.Module):
    """AlexNet feature extractor with leaky integrated blocks."""

    def __init__(self, tau_schedule_ms: Iterable[float] | float, delta_t_ms: float, learnable_alpha: bool = False):
        super().__init__()
        model = alexnet(weights=None)
        blocks: List[nn.Module] = []
        current = []
        for layer in model.features:
            current.append(layer)
            if isinstance(layer, nn.ReLU):
                blocks.append(nn.Sequential(*current))
                current = []
        if isinstance(tau_schedule_ms, (int, float)):
            tau_schedule_ms = [float(tau_schedule_ms)] * len(blocks)
        assert len(tau_schedule_ms) == len(blocks)
        li_layers = []
        for tau, block in zip(tau_schedule_ms, blocks):
            li_layers.append(LIConv2d(block, tau, delta_t_ms))
        self.li_layers = nn.ModuleList(li_layers)

    def forward(self, video: torch.Tensor) -> List[torch.Tensor]:
        x = video
        activations: List[torch.Tensor] = []
        for li in self.li_layers:
            x = li(x)
            activations.append(x)
        return activations


def _compute_rai(act: torch.Tensor) -> np.ndarray:
    """Compute Response Adaptation Index for a single activation tensor."""
    act_mean = act.mean(dim=(3, 4))  # (B,T,C)
    r1 = act_mean[:, 0]
    r2 = act_mean[:, 7:10].mean(dim=1)
    rai = (r1 - r2) / (r1 + r2 + 1e-8)
    return rai.cpu().numpy()


def analyze_network(dataset_path: str = "dataset.pt",
                    results_dir: str = "results",
                    num_clips: int = 20,
                    clip_len: int = 120,
                    image_size: int = 224,
                    fps: int = 30) -> None:
    """Feedforward the dataset through LICNN and analyse adaptation."""
    os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}, generating a new one...")
        create_dataset_file(dataset_path, num_clips, clip_len, image_size, fps)

    data = torch.load(dataset_path)  # (N,T,3,H,W)
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    net = LICNN(tau_schedule_ms=[50, 50, 50, 150, 150], delta_t_ms=1000 / fps)
    net.eval()

    all_rais = [ [] for _ in range(len(net.li_layers)) ]
    for videos in loader:
        with torch.no_grad():
            activations = net(videos)
        for i, act in enumerate(activations):
            rai = _compute_rai(act)
            all_rais[i].append(rai)

    summary_lines = []
    for idx, rai_list in enumerate(all_rais):
        rai_vals = np.concatenate(rai_list, axis=0).reshape(-1)
        np.save(os.path.join(results_dir, f"rai_layer{idx+1}.npy"), rai_vals)
        plt.figure()
        plt.hist(rai_vals, bins=50)
        plt.xlabel("RAI")
        plt.ylabel("count")
        plt.title(f"Layer {idx+1} RAI distribution")
        plt.savefig(os.path.join(results_dir, f"rai_hist_layer{idx+1}.png"))
        plt.close()
        adaptation_frac = (rai_vals < 0).mean()
        summary_lines.append(f"Layer {idx+1}: {adaptation_frac:.3f} showing adaptation\n")

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.writelines(summary_lines)


if __name__ == "__main__":
    analyze_network()
