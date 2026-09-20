"""Qualitative visualization helpers for TMPA-compatible remote-sensing evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.imagecorruptions import corrupt


def should_save_vis(sample_idx: int, interval: int) -> bool:
    """Return True for deterministic interval sampling, starting from sample 0."""
    if interval <= 0:
        raise ValueError(f"vis_interval must be > 0, got {interval}")
    return int(sample_idx) % int(interval) == 0


def _to_mask_numpy(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask, got shape {mask.shape}")
    return mask.astype(np.int64, copy=False)


def colorize_mask(mask, palette, ignore_index: int = 255) -> np.ndarray:
    """Colorize class ids with the dataset palette; ignore pixels are white."""
    mask_np = _to_mask_numpy(mask)
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        rgb[mask_np == class_idx] = np.asarray(color, dtype=np.uint8)
    rgb[mask_np == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _save_label(mask, path: Path) -> None:
    mask_np = _to_mask_numpy(mask)
    if mask_np.size and (mask_np.min() < 0 or mask_np.max() > 255):
        Image.fromarray(mask_np.astype(np.int32), mode="I").save(path)
    else:
        Image.fromarray(mask_np.astype(np.uint8), mode="L").save(path)


def _apply_remote_corruption(clean_rgb: np.ndarray, corruption: str, severity: int, sample_idx: int) -> np.ndarray:
    """Reproduce the exact TMPA-compatible DAF corruption in RGB space."""
    if corruption == "original":
        return clean_rgb.copy()

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(sample_idx))
        corrupted = corrupt(
            clean_rgb,
            severity=int(severity),
            corruption_name=corruption,
        )
    finally:
        np.random.set_state(rng_state)
    return np.asarray(corrupted, dtype=np.uint8)


def save_remote_visualization(
    *,
    vis_root: str,
    dataset: str,
    corruption: str,
    severity: int,
    sample_idx: int,
    image_path: str,
    pred,
    gt,
    palette,
    method_name: str,
    ignore_index: int = 255,
) -> str:
    """Save clean/corrupted input, raw labels, and colorized GT/prediction."""
    image_path = str(image_path)
    sample_name = Path(image_path).stem
    sample_dir = (
        Path(vis_root)
        / str(dataset)
        / str(corruption)
        / f"{int(sample_idx):06d}_{sample_name}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    corrupted_rgb = _apply_remote_corruption(
        clean_rgb, corruption, severity, sample_idx
    )

    Image.fromarray(clean_rgb, mode="RGB").save(sample_dir / "clean.png")
    Image.fromarray(corrupted_rgb, mode="RGB").save(sample_dir / "corrupted.png")

    _save_label(gt, sample_dir / "gt_label.png")
    _save_label(pred, sample_dir / "prediction_label.png")
    Image.fromarray(
        colorize_mask(gt, palette, ignore_index=ignore_index), mode="RGB"
    ).save(sample_dir / "gt_vis.png")
    Image.fromarray(
        colorize_mask(pred, palette, ignore_index=ignore_index), mode="RGB"
    ).save(sample_dir / "prediction_vis.png")

    metadata = {
        "dataset": str(dataset),
        "corruption": str(corruption),
        "corruption_severity": int(severity),
        "sample_idx": int(sample_idx),
        "image_name": sample_name,
        "image_path": image_path,
        "method": str(method_name),
    }
    with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return os.fspath(sample_dir)
