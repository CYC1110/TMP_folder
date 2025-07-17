import math
import random
from typing import Tuple

import numpy as np
import torch


def _generate_grating(frame_size: int,
                      spatial_freq: float,
                      orientation_deg: float,
                      phase: float = 0.0,
                      contrast: float = 1.0) -> np.ndarray:
    """Return a single grayscale sinusoidal grating image in [-1, 1]."""
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
    """Generate a (T,H,W) clip of a drifting grating."""
    T = int(duration_s * fps)
    phase_speed = 2 * math.pi / T
    frames = []
    for t in range(T):
        phase = t * phase_speed
        img = _generate_grating(frame_size, spatial_freq, orientation_deg, phase, contrast)
        frames.append(img)
    return np.stack(frames, axis=0)


def make_adaptation_oddball_clip(frame_size: int = 224,
                                 fps: int = 60,
                                 n_adapt_frames: int = 100,
                                 n_test_frames: int = 1,
                                 orientation_A: float = 0.0,
                                 orientation_B: float = 90.0,
                                 spatial_freq: float = 3.0,
                                 contrast: float = 1.0) -> np.ndarray:
    """Return an A...AB oddball sequence."""
    frames_A = make_drifting_grating_clip(frame_size, spatial_freq, orientation_A,
                                          contrast, fps, duration_s=n_adapt_frames / fps)
    frame_B = _generate_grating(frame_size, spatial_freq, orientation_B, phase=0.0, contrast=contrast)
    frames = list(frames_A)
    frames.append(frame_B)
    for _ in range(int(0.5 * fps)):
        frames.append(_generate_grating(frame_size, spatial_freq, orientation_A, phase=0.0, contrast=contrast))
    return np.stack(frames, axis=0)


def _to_rgb(frames: np.ndarray) -> np.ndarray:
    frames = (frames + 1.0) / 2.0
    rgb = np.repeat(frames[:, None, :, :], 3, axis=1)
    return rgb.astype(np.float32)


class VisualAdaptationDataset(torch.utils.data.Dataset):
    """On-the-fly generated dataset of adaptation stimuli."""

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

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_clips

    def _random_grating_params(self) -> Tuple[float, float, float]:
        orientation = self.rng.uniform(0, 180)
        spatial_freq = self.rng.uniform(1.0, 6.0)
        contrast = self.rng.choice([0.1, 0.3, 0.5, 0.7, 1.0])
        return orientation, spatial_freq, contrast

    def __getitem__(self, idx: int) -> torch.Tensor:
        stim_type = idx % 3
        if stim_type == 0:
            ori, sf, con = self._random_grating_params()
            clip = make_drifting_grating_clip(self.image_size, sf, ori, con,
                                              fps=self.fps,
                                              duration_s=self.clip_len_frames / self.fps)
        elif stim_type == 1:
            ori, sf, _ = self._random_grating_params()
            high = make_drifting_grating_clip(self.image_size, sf, ori, 1.0,
                                              fps=self.fps,
                                              duration_s=(self.clip_len_frames // 2) / self.fps)
            low = make_drifting_grating_clip(self.image_size, sf, ori, 0.1,
                                             fps=self.fps,
                                             duration_s=(self.clip_len_frames // 2) / self.fps)
            clip = np.concatenate([high, low], axis=0)
        else:
            ori_A, sf, con = self._random_grating_params()
            ori_B = (ori_A + 90) % 180
            clip = make_adaptation_oddball_clip(self.image_size, self.fps,
                                                n_adapt_frames=self.clip_len_frames - 60,
                                                n_test_frames=1,
                                                orientation_A=ori_A,
                                                orientation_B=ori_B,
                                                spatial_freq=sf,
                                                contrast=con)
        clip = clip[:self.clip_len_frames]
        if clip.shape[0] < self.clip_len_frames:
            pad = np.zeros((self.clip_len_frames - clip.shape[0], *clip.shape[1:]), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)
        clip_rgb = _to_rgb(clip)
        return torch.from_numpy(clip_rgb)


def create_dataset_file(path: str,
                        num_clips: int = 20,
                        clip_len_frames: int = 120,
                        image_size: int = 64,
                        fps: int = 30,
                        seed: int | None = None) -> None:
    """Generate a dataset on disk for reuse."""
    ds = VisualAdaptationDataset(num_clips, clip_len_frames, image_size, fps, seed)
    data = torch.stack([ds[i] for i in range(num_clips)], dim=0)
    torch.save(data, path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and save adaptation dataset")
    parser.add_argument("--output", type=str, default="dataset.pt")
    parser.add_argument("--num_clips", type=int, default=20)
    parser.add_argument("--clip_len", type=int, default=120)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    create_dataset_file(args.output, args.num_clips, args.clip_len, args.image_size, args.fps)
