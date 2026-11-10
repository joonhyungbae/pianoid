#!/usr/bin/env python3
"""
Hand + ARIA Dataset (Explore 02).

MIDI + optional ARIA embedding for input conditioning sweep.
Returns: hand, midi_features, aria_embed (B, 512) per window.
Run from project root or with PYTHONPATH including project root.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.constants import LWRIST_IDX, LHAND_IDX, RWRIST_IDX, RHAND_IDX

# ARIA temporal alignment: s13 uses 60fps, window_sec=8 -> 480 frames, stride 240
ARIA_STRIDE_FRAMES = 240


def _resolve_cache_dir(cache_dir: str) -> Path:
    p = Path(cache_dir)
    if not p.is_absolute():
        p = ROOT / p
    return p


class HandAriaDataset(Dataset):
    """
    Hand + MIDI + ARIA dataset for input conditioning sweep.
    Returns aria_embed per window: (512,) from temporal index or global.
    """
    def __init__(
        self,
        cache_dir: str = "MOSA_dataset/training_cache",
        split: str = "train",
        window_size: int = 480,
        stride: int = 240,
        preload: bool = True,
    ):
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.preload = preload

        with open(self.cache_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.split_samples = self.metadata["splits"][split]["samples"]
        self._cache = {}
        if self.preload:
            self._preload_data()
        self.windows = []
        self._create_windows()
        print(f"[{split}] HandAriaDataset: {len(self.windows)} windows from {len(self.split_samples)} samples")

    def _preload_data(self):
        split_dir = self.cache_dir / self.split
        print(f"Preloading {len(self.split_samples)} files into memory...")
        for enum_idx, sample in enumerate(tqdm(self.split_samples, desc=f"Loading {self.split}")):
            npz_path = split_dir / f"{sample['idx']:05d}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            joints = data["joints"].astype(np.float32)
            hand = np.concatenate([
                joints[:, LWRIST_IDX], joints[:, LHAND_IDX],
                joints[:, RWRIST_IDX], joints[:, RHAND_IDX],
            ], axis=-1)
            velocity = data.get("onset_density_per_key", np.zeros((len(joints), 88))).astype(np.float32)
            onset = data.get("key_onset", np.zeros((len(joints), 88))).astype(np.float32)
            piano_roll = (onset > 0).astype(np.float32)
            if "active_keys" in data:
                active_keys = data["active_keys"]
                for t in range(len(piano_roll)):
                    for k in active_keys[t]:
                        if 0 <= k < 88:
                            piano_roll[t, k] = 1.0
            aria_embed = data.get("aria_embed")
            if aria_embed is not None:
                aria_embed = aria_embed.astype(np.float32)
                if aria_embed.ndim == 1:
                    pass  # (512,) global
                else:
                    pass  # (num_windows, 512) temporal
            self._cache[enum_idx] = {
                "hand": hand, "piano_roll": piano_roll,
                "velocity": velocity, "onset": onset,
                "num_frames": sample["num_frames"],
                "aria_embed": aria_embed,
            }

    def _get_sample_data(self, sample_idx: int) -> Dict:
        if sample_idx in self._cache:
            return self._cache[sample_idx]
        sample = self.split_samples[sample_idx]
        split_dir = self.cache_dir / self.split
        data = np.load(split_dir / f"{sample['idx']:05d}.npz")
        joints = data["joints"].astype(np.float32)
        hand = np.concatenate([
            joints[:, LWRIST_IDX], joints[:, LHAND_IDX],
            joints[:, RWRIST_IDX], joints[:, RHAND_IDX],
        ], axis=-1)
        velocity = data.get("onset_density_per_key", np.zeros((len(joints), 88))).astype(np.float32)
        onset = data.get("key_onset", np.zeros((len(joints), 88))).astype(np.float32)
        piano_roll = (onset > 0).astype(np.float32)
        if "active_keys" in data:
            for t in range(len(piano_roll)):
                for k in data["active_keys"][t]:
                    if 0 <= k < 88:
                        piano_roll[t, k] = 1.0
        aria_embed = data.get("aria_embed")
        if aria_embed is not None:
            aria_embed = aria_embed.astype(np.float32)
        return {
            "hand": hand, "piano_roll": piano_roll,
            "velocity": velocity, "onset": onset,
            "num_frames": sample["num_frames"],
            "aria_embed": aria_embed,
        }

    def _create_windows(self):
        for sample_idx, sample in enumerate(self.split_samples):
            num_frames = sample["num_frames"]
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                self.windows.append({"sample_idx": sample_idx, "start": start, "end": start + self.window_size})
            if num_frames >= self.window_size:
                last_start = num_frames - self.window_size
                if last_start % self.stride != 0:
                    self.windows.append({"sample_idx": sample_idx, "start": last_start, "end": num_frames})

    def __len__(self):
        return len(self.windows)

    def _get_aria_for_window(self, aria_embed, start: int) -> np.ndarray:
        """Return (512,) for this window. Temporal: index by start//stride; global: same vector."""
        if aria_embed is None:
            return np.zeros(512, dtype=np.float32)
        if aria_embed.ndim == 1:
            return aria_embed
        # (num_windows, 512)
        wi = start // ARIA_STRIDE_FRAMES
        wi = min(wi, aria_embed.shape[0] - 1)
        return aria_embed[wi]

    def __getitem__(self, idx):
        w = self.windows[idx]
        d = self._get_sample_data(w["sample_idx"])
        s, e = w["start"], w["end"]
        aria = self._get_aria_for_window(d.get("aria_embed"), w["start"])
        return {
            "hand": torch.tensor(d["hand"][s:e], dtype=torch.float32),
            "midi_features": {
                "piano_roll": torch.tensor(d["piano_roll"][s:e], dtype=torch.float32),
                "velocity": torch.tensor(d["velocity"][s:e], dtype=torch.float32),
                "onset": torch.tensor(d["onset"][s:e], dtype=torch.float32),
            },
            "aria_embed": torch.tensor(aria, dtype=torch.float32),
        }


def get_hand_aria_loader(
    cache_dir: str = "MOSA_dataset/training_cache",
    split: str = "train",
    batch_size: int = 32,
    window_size: int = 480,
    stride: int = 240,
    num_workers: int = 4,
    shuffle: bool = True,
    preload: bool = True,
):
    from torch.utils.data import DataLoader
    cache_dir = str(_resolve_cache_dir(cache_dir))
    dataset = HandAriaDataset(
        cache_dir=cache_dir, split=split,
        window_size=window_size, stride=stride, preload=preload,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
    )


if __name__ == "__main__":
    dataset = HandAriaDataset(
        cache_dir=str(ROOT / "MOSA_dataset" / "training_cache"),
        split="train", window_size=480, stride=240, preload=True,
    )
    print("Length:", len(dataset))
    s = dataset[0]
    print("Sample keys:", s["hand"].shape, s["midi_features"]["piano_roll"].shape, s["aria_embed"].shape)
    print("OK")
