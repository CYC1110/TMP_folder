# -*- coding: utf-8 -*-
"""li_cnn_and_dataset.py

This single file contains:
1. A **Leaky‑Integrator Convolutional Neural Network (LI‑CNN)** implementation that wraps
   the convolutional blocks of AlexNet with an exponential leaky‑integrator so that
   each feature map exhibits temporal adaptation dynamics similar to those observed
   in visual‑cortex neurons.
2. Utility functions and a lightweight PyTorch **Dataset** for generating synthetic
   spatio‑temporal stimuli commonly used in visual‐adaptation experiments (drifting
   gratings of varying contrast/orientation, an adaptation oddball protocol, and a
   generic natural‑movie placeholder).

Both parts are self‑contained: import the file, construct the dataset, then feed video
batches to the network to analyse emergent adaptation indices.

Usage example (put in a separate script or notebook)::

    from li_cnn_and_dataset import LICNN, VisualAdaptationDataset
    import torch

    # Build network — 224×224 input, 60 Hz frame rate, τ=50 ms for early layers
    net = LICNN(tau_schedule_ms=[50, 50, 50, 150, 150], delta_t_ms=1000/60)
    net.eval()  # we keep it untrained / randomly‑initialised

    # Make a tiny dataset of 8 video clips (B,T,C,H,W) = (8,300,3,224,224)
    ds = VisualAdaptationDataset(num_clips=8,
                                 clip_len_frames=300,
                                 image_size=224,
                                 fps=60)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    for video_batch in loader:
        with torch.no_grad():
            # forward returns per‑time‑step activations from each LI layer
            activations = net(video_batch)  # list[ Tensor(B,T,channels,H,W) ]
            # ← Compute adaptation metrics (RAI, T½, …) from `activations`
            break
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

# -----------------------------------------------------------------------------
# 1. Leaky‑Integrator modules
# -----------------------------------------------------------------------------

class ExponentialLeakyIntegrator(nn.Module):
    """Implements the discrete‑time recursion

        ṽ_t = x_t + α · ṽ_{t‑1},   α = exp(‑Δt/τ)

    where x_t is the instantaneous activation and ṽ_t the leaky‑integrated signal.
    """

    def __init__(self, tau_ms: float, delta_t_ms: float, learnable_alpha: bool = False):
        super().__init__()
        alpha_val = math.exp(-delta_t_ms / tau_ms)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_val, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_val, dtype=torch.float32))

    def forward(self, x_t: torch.Tensor, prev_state: torch.Tensor | None) -> torch.Tensor:
        """x_t: (B, C, H, W) instantaneous activation at time t
            prev_state: integrated activation at t‑1, same shape or None (treated as 0)
        """
        if prev_state is None:
            prev_state = torch.zeros_like(x_t)
        return x_t + self.alpha * prev_state


class LIConv2d(nn.Module):
    """Wraps a Conv‑BN‑ReLU trio (or just Conv‑ReLU) with a leaky integrator.

    The layer expects an input tensor of shape **(B, T, C, H, W)** and returns the
    leaky‑integrated activations of the same shape (re‑time‑stacked).
    """

    def __init__(self, conv_block: nn.Module, tau_ms: float, delta_t_ms: float):
        super().__init__()
        self.conv_block = conv_block  # e.g. nn.Sequential(Conv2d, ReLU)
        self.integrator = ExponentialLeakyIntegrator(tau_ms, delta_t_ms)

    @torch.jit.script_method  # TorchScript for a minor speedup; comment out if tracing fails
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,C,H,W)
        B, T, C, H, W = x.shape
        outputs = []
        prev_state: torch.Tensor | None = None
        for t in range(T):
            frame = x[:, t]           # (B,C,H,W)
            y_t = self.conv_block(frame)
            v_t = self.integrator(y_t, prev_state)
            outputs.append(v_t)
            prev_state = v_t
        return torch.stack(outputs, dim=1)  # (B,T,C′,H′,W′)


class LICNN(nn.Module):
    """AlexNet‑style CNN where every convolutional block is leaky‑integrated.

    Parameters
    ----------
    tau_schedule_ms : List[float]
        One τ per convolutional block (AlexNet has 5).  If a single float is given
        it is broadcast to all layers.
    delta_t_ms : float
        Video frame interval used when the network is fed sequential frames.
    learnable_alpha : bool
        Whether α is a free parameter per layer.
    """

    def __init__(self, tau_schedule_ms: Iterable[float] | float,
                 delta_t_ms: float,
                 learnable_alpha: bool = False):
        super().__init__()

        # AlexNet feature extractor (5 Conv blocks)
        model = alexnet(weights=None)  # random / untrained
        feature_layers: List[nn.Module] = []
        current_block = []
        conv_count = 0
        for layer in model.features:
            current_block.append(layer)
            if isinstance(layer, nn.ReLU):  # AlexNet pattern Conv→BN?→ReLU
                feature_layers.append(nn.Sequential(*current_block))
                current_block = []
                conv_count += 1
        if current_block:
            feature_layers.append(nn.Sequential(*current_block))

        if isinstance(tau_schedule_ms, (int, float)):
            tau_schedule_ms = [float(tau_schedule_ms)] * len(feature_layers)
        assert len(tau_schedule_ms) == len(feature_layers), "τ schedule must match # conv blocks"

        li_layers = []
        for tau, block in zip(tau_schedule_ms, feature_layers):
            li_layers.append(LIConv2d(block, tau_ms=tau, delta_t_ms=delta_t_ms))
        self.li_layers = nn.ModuleList(li_layers)

    def forward(self, video: torch.Tensor) -> List[torch.Tensor]:
        """video: (B,T,3,H,W) Returns list of activations at each LI layer."""
        x = video
        activations: List[torch.Tensor] = []
        for li in self.li_layers:
            x = li(x)
            activations.append(x)  # (B,T,C,H,W)
            # Down‑sample in AlexNet after certain layers; replicate with max‑pool
            # Here we imitate original Frances conv layout: pool after layers 1,2,5
            # For simplicity, apply 2×2 stride‑2 max‑pool after each li layer *except* the last
            if li is not self.li_layers[-1]:
                B, T, C, H, W = x.shape
                x = F.max_pool2d(x.view(B * T, C, H, W), kernel_size=2, stride=2)
                x = x.view(B, T, C, H // 2, W // 2)
        return activations

# -----------------------------------------------------------------------------
# 2. Synthetic spatio‑temporal stimulus generation
# -----------------------------------------------------------------------------

def _generate_grating(frame_size: int,
                      spatial_freq: float,
                      orientation_deg: float,
                      phase: float = 0.0,
                      contrast: float = 1.0) -> np.ndarray:
    """Returns a single grayscale sinusoidal grating image in the range [‑1,1]."""
    # Build coordinate system
    radians = math.radians(orientation_deg)
    fx = spatial_freq * math.cos(radians)
    fy = spatial_freq * math.sin(radians)
    ys, xs = np.linspace(-1, 1, frame_size), np.linspace(-1, 1, frame_size)
    X, Y = np.meshgrid(xs, ys)
    grating = np.cos(2 * math.pi * (fx * X + fy * Y) + phase)
    return contrast * grating.astype(np.float32)


def make_drifting_grating_clip(frame_size: int = 224,
                               spatial_freq: float = 3.0,
                               orientation_deg: float = 0.0,
                               contrast: float = 1.0,
                               fps: int = 60,
                               duration_s: float = 5.0) -> np.ndarray:
    """Generates a (T,H,W) clip of a drifting grating with constant phase speed."""
    T = int(duration_s * fps)
    phase_speed = 2 * math.pi / T  # one full cycle during clip
    frames = []
    for t in range(T):
        phase = t * phase_speed
        img = _generate_grating(frame_size, spatial_freq, orientation_deg, phase, contrast)
        frames.append(img)
    return np.stack(frames, axis=0)  # (T,H,W)


def make_adaptation_oddball_clip(frame_size: int = 224,
                                 fps: int = 60,
                                 n_adapt_frames: int = 100,
                                 n_test_frames: int = 1,
                                 orientation_A: float = 0.0,
                                 orientation_B: float = 90.0,
                                 spatial_freq: float = 3.0,
                                 contrast: float = 1.0) -> np.ndarray:
    """A‑A‑A‑…‑A‑B‑A style oddball sequence for SSA/rapid adaptation measures."""
    frames_A = make_drifting_grating_clip(frame_size, spatial_freq, orientation_A,
                                          contrast, fps, duration_s=n_adapt_frames / fps)
    frame_B = _generate_grating(frame_size, spatial_freq, orientation_B, phase=0.0, contrast=contrast)
    frames = list(frames_A)
    # Insert B at the end of adaptation block
    frames.append(frame_B)
    # Add a short tail of A to look at recovery
    for _ in range(int(0.5 * fps)):
        frames.append(_generate_grating(frame_size, spatial_freq, orientation_A, phase=0.0, contrast=contrast))
    return np.stack(frames, axis=0)


def _to_rgb(frames: np.ndarray) -> np.ndarray:
    """Converts (T,H,W) grayscale in [‑1,1] → (T,3,H,W) RGB float32 in [0,1]."""
    frames = (frames + 1.0) / 2.0  # [‑1,1]→[0,1]
    rgb = np.repeat(frames[:, None, :, :], 3, axis=1)
    return rgb.astype(np.float32)


class VisualAdaptationDataset(torch.utils.data.Dataset):
    """On‑the‑fly generated dataset of adaptation stimuli for a randomly‑initialised CNN.

    Each call returns a video tensor **(T,3,H,W)** in *float32* range [0,1].
    """

    def __init__(self,
                 num_clips: int = 100,
                 clip_len_frames: int = 300,
                 image_size: int = 224,
                 fps: int = 60,
                 seed: int | None = None):
        self.num_clips = num_clips
        self.clip_len_frames = clip_len_frames
        self.image_size = image_size
        self.fps = fps
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_clips

    def _random_grating_params(self) -> Tuple[float, float, float]:
        orientation = self.rng.uniform(0, 180)
        spatial_freq = self.rng.uniform(1.0, 6.0)  # cycles per degree (arbitrary)
        contrast = self.rng.choice([0.1, 0.3, 0.5, 0.7, 1.0])
        return orientation, spatial_freq, contrast

    def __getitem__(self, idx):
        # Alternate among stimulus types
        stim_type = idx % 3
        if stim_type == 0:
            # Drifting grating
            ori, sf, con = self._random_grating_params()
            clip = make_drifting_grating_clip(self.image_size, sf, ori, con,
                                              fps=self.fps,
                                              duration_s=self.clip_len_frames / self.fps)
        elif stim_type == 1:
            # Contrast adaptation: high‑contrast → low‑contrast block
            ori, sf, _ = self._random_grating_params()
            high = make_drifting_grating_clip(self.image_size, sf, ori, 1.0,
                                              fps=self.fps,
                                              duration_s=(self.clip_len_frames // 2) / self.fps)
            low = make_drifting_grating_clip(self.image_size, sf, ori, 0.1,
                                             fps=self.fps,
                                             duration_s=(self.clip_len_frames // 2) / self.fps)
            clip = np.concatenate([high, low], axis=0)
        else:
            # Oddball adaptation
            ori_A, sf, con = self._random_grating_params()
            ori_B = (ori_A + 90) % 180  # orthogonal
            clip = make_adaptation_oddball_clip(self.image_size, self.fps,
                                                n_adapt_frames=self.clip_len_frames - 60,
                                                n_test_frames=1,
                                                orientation_A=ori_A,
                                                orientation_B=ori_B,
                                                spatial_freq=sf,
                                                contrast=con)
        # Ensure length
        clip = clip[:self.clip_len_frames]
        if clip.shape[0] < self.clip_len_frames:  # pad with zeros if needed
            pad = np.zeros((self.clip_len_frames - clip.shape[0], *clip.shape[1:]), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)
        clip_rgb = _to_rgb(clip)  # (T,3,H,W)
        return torch.from_numpy(clip_rgb)

# End of file
