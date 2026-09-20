"""SegEarth-OV support for the shared DAF OVSS backbone.

This module keeps DAF adaptation methods unchanged while reproducing the
SegEarth-OV visual head: SegEarth attention, frozen JBU feature upsampling,
dataset-specific class-token fusion, and confidence post-processing metadata.
"""

import math
import os
import urllib.request
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SEGEARTH_CKPT_URL = (
    "https://raw.githubusercontent.com/likyoo/SegEarth-OV/main/"
    "simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt"
)
SEGEARTH_CKPT_NAME = "xclip_jbu_one_million_aid.ckpt"

_DEFAULT_CFG = {
    "prob_thd": 0.0,
    "cls_token_lambda": -0.30,
    "logit_scale": 50.0,
    "bg_idx": 0,
}

_DATASET_CFG = {
    "openearthmap": {"prob_thd": 0.00, "cls_token_lambda": -0.35, "logit_scale": 50.0, "bg_idx": 0},
    "loveda": {"prob_thd": 0.10, "cls_token_lambda": -0.30, "logit_scale": 40.0, "bg_idx": 0},
    "isaid": {"prob_thd": 0.20, "cls_token_lambda": -0.00, "logit_scale": 50.0, "bg_idx": 0},
    "potsdam": {"prob_thd": 0.00, "cls_token_lambda": -0.10, "logit_scale": 50.0, "bg_idx": 5},
    "vaihingen": {"prob_thd": 0.00, "cls_token_lambda": -0.48, "logit_scale": 48.0, "bg_idx": 5},
    "uavid": {"prob_thd": 0.05, "cls_token_lambda": -0.30, "logit_scale": 45.0, "bg_idx": 0},
    "udd5": {"prob_thd": 0.00, "cls_token_lambda": -0.20, "logit_scale": 20.0, "bg_idx": 4},
    "vdd": {"prob_thd": 0.00, "cls_token_lambda": -0.30, "logit_scale": 45.0, "bg_idx": 0},
    "whu_aerial": {"prob_thd": 0.19, "cls_token_lambda": -0.00, "logit_scale": 35.0, "bg_idx": 0},
    "whu_sat": {"prob_thd": 0.30, "cls_token_lambda": -0.45, "logit_scale": 50.0, "bg_idx": 0},
    "inria": {"prob_thd": 0.20, "cls_token_lambda": -0.10, "logit_scale": 35.0, "bg_idx": 0},
    "xbd": {"prob_thd": 0.10, "cls_token_lambda": -0.30, "logit_scale": 25.0, "bg_idx": 0},
    "chn6-cug": {"prob_thd": 0.10, "cls_token_lambda": -0.50, "logit_scale": 40.0, "bg_idx": 0},
    "deepglobe": {"prob_thd": 0.10, "cls_token_lambda": -0.40, "logit_scale": 40.0, "bg_idx": 0},
    "massachusetts": {"prob_thd": 0.10, "cls_token_lambda": -0.35, "logit_scale": 20.0, "bg_idx": 0},
    "spacenet": {"prob_thd": 0.39, "cls_token_lambda": -0.30, "logit_scale": 50.0, "bg_idx": 0},
    "wbs_si": {"prob_thd": 0.10, "cls_token_lambda": -0.50, "logit_scale": 25.0, "bg_idx": 0},
}

_DATASET_ALIASES = {
    "lovedadataset": "loveda",
}


try:
    from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv
except Exception:
    AdaptiveConv = None


def _canonical_dataset(dataset):
    if dataset is None:
        return None
    key = str(dataset).lower()
    return _DATASET_ALIASES.get(key, key)


def get_segearth_config(dataset=None):
    cfg = dict(_DEFAULT_CFG)
    key = _canonical_dataset(dataset)
    if key in _DATASET_CFG:
        cfg.update(_DATASET_CFG[key])
    return cfg


def _adaptive_conv_fallback(source, filters):
    """Memory-stable PyTorch fallback for FeatUp AdaptiveConv.

    It is slower than the CUDA extension but preserves the JBU operation and
    keeps the branch runnable when featup's custom CUDA op is unavailable.
    """
    _, out_h, out_w, k_h, k_w = filters.shape
    output = source.new_zeros((source.shape[0], source.shape[1], out_h, out_w))
    for y in range(k_h):
        for x in range(k_w):
            patch = source[:, :, y:y + out_h, x:x + out_w]
            weight = filters[:, :, :, y, x].unsqueeze(1).to(source.dtype)
            output.add_(patch * weight)
    return output


class JBULearnedRange(nn.Module):
    def __init__(self, guidance_dim, feat_dim, key_dim, radius=5):
        super().__init__()
        self.radius = radius
        self.diameter = radius * 2 + 1
        self.key_dim = key_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = nn.Sequential(
            nn.Conv2d(guidance_dim, key_dim, 1, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(key_dim, key_dim, 1, 1),
        )
        self.fixup_proj = nn.Sequential(
            nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )
        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, guidance):
        b, _, h, w = guidance.shape
        proj = self.range_proj(guidance)
        proj_pad = F.pad(proj, pad=[self.radius] * 4, mode="reflect")
        queries = (
            nn.Unfold(self.diameter)(proj_pad)
            .reshape(b, self.key_dim, self.diameter * self.diameter, h, w)
            .permute(0, 1, 3, 4, 2)
        )
        pos_temp = self.range_temp.exp().clamp(1e-4, 1e4)
        return F.softmax(
            pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj),
            dim=1,
        )

    def get_spatial_kernel(self, device, dtype):
        axis = torch.linspace(-1, 1, self.diameter, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        patch = torch.stack([yy, xx], dim=0)
        sigma = self.sigma_spatial.to(device=device, dtype=dtype)
        return torch.exp(-patch.square().sum(0) / (2 * sigma ** 2)).reshape(
            1, self.diameter * self.diameter, 1, 1
        )

    def forward(self, source, guidance):
        b, _, h, w = guidance.shape
        spatial_kernel = self.get_spatial_kernel(source.device, source.dtype)
        range_kernel = self.get_range_kernel(guidance).to(source.dtype)

        combined = range_kernel * spatial_kernel
        combined = combined / combined.sum(1, keepdim=True).clamp_min(1e-7)
        combined = combined + 0.1 * self.fixup_proj(
            torch.cat([combined, guidance], dim=1)
        ).to(combined.dtype)
        combined = (
            combined.permute(0, 2, 3, 1)
            .reshape(b, h, w, self.diameter, self.diameter)
        )

        hr_source = F.interpolate(
            source, size=(h, w), mode="bicubic", align_corners=False
        )
        hr_source = F.pad(hr_source, pad=[self.radius] * 4, mode="reflect")

        if AdaptiveConv is not None:
            return AdaptiveConv.apply(hr_source, combined.to(hr_source.dtype))
        return _adaptive_conv_fallback(hr_source, combined)


class JBUOne(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.up = JBULearnedRange(3, feat_dim, 32, radius=5)
        self.fixup_proj = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
        )

    def _upsample(self, source, guidance):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return self.up(source, small_guidance)

    def forward(self, source, guidance):
        source_2 = self._upsample(source, guidance)
        source_4 = self._upsample(source_2, guidance)
        source_8 = self._upsample(source_4, guidance)
        source_16 = self._upsample(source_8, guidance)
        return self.fixup_proj(source_16) * 0.1 + source_16


def _checkpoint_path():
    override = os.environ.get("SEGEARTH_UPSAMPLER_CKPT")
    if override:
        return Path(override).expanduser()

    # Reuse the checkpoint from a local SegEarth-OV/TMPA checkout when present.
    for candidate in (
        Path("simfeatup_dev/weights") / SEGEARTH_CKPT_NAME,
        Path("../TMPA/simfeatup_dev/weights") / SEGEARTH_CKPT_NAME,
        Path("../SegEarth-OV/simfeatup_dev/weights") / SEGEARTH_CKPT_NAME,
    ):
        if candidate.is_file():
            return candidate.resolve()

    cache_root = Path(
        os.environ.get(
            "SEGEARTH_CACHE_ROOT",
            str(Path.home() / ".cache" / "segearth-ov"),
        )
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / SEGEARTH_CKPT_NAME


def _ensure_checkpoint(path):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SEGEARTH_CKPT_URL, str(path))
    except Exception as exc:
        raise RuntimeError(
            "Unable to obtain the official SegEarth-OV JBU checkpoint. "
            "Download xclip_jbu_one_million_aid.ckpt from likyoo/SegEarth-OV "
            "and set SEGEARTH_UPSAMPLER_CKPT to its local path."
        ) from exc
    return path


def build_segearth_upsampler(feat_dim, device, dtype):
    ckpt_path = _ensure_checkpoint(_checkpoint_path())
    upsampler = JBUOne(feat_dim)

    payload = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("upsampler."):
            key = key[len("upsampler."):]
        elif key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    try:
        upsampler.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        # The official SegEarth-OV loader strips the first 10 characters
        # from FeatUp checkpoint keys. Keep that exact fallback for checkpoint
        # variants whose prefix is not literally "upsampler.".
        stripped = {
            key[10:]: value
            for key, value in state_dict.items()
            if len(key) > 10
        }
        try:
            upsampler.load_state_dict(stripped, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Invalid SegEarth-OV JBU checkpoint: {ckpt_path}"
            ) from exc

    upsampler = upsampler.to(device=device, dtype=dtype)
    upsampler.requires_grad_(False)
    upsampler.eval()

    if AdaptiveConv is None:
        warnings.warn(
            "featup AdaptiveConv CUDA extension is unavailable; using a slower "
            "PyTorch fallback for SegEarth-OV JBU upsampling.",
            RuntimeWarning,
        )
    return upsampler


def configure_segearth_model(model, dataset=None, device="cpu"):
    cfg = get_segearth_config(dataset)

    model.segearth_enabled = True
    model.segearth_feature_up = os.environ.get(
        "SEGEARTH_FEATURE_UP", "1"
    ).lower() not in {"0", "false", "no"}
    model.segearth_cls_token_lambda = float(cfg["cls_token_lambda"])
    model.segearth_prob_thd = float(cfg["prob_thd"])
    model.segearth_bg_idx = int(cfg["bg_idx"])
    model.segearth_dataset = _canonical_dataset(dataset)

    # SegEarth-OV uses a fixed similarity scale instead of CLIP's learned scale.
    with torch.no_grad():
        model.logit_scale.fill_(math.log(float(cfg["logit_scale"])))
    model.logit_scale.requires_grad_(False)

    if model.segearth_feature_up:
        feat_dim = int(model.text_projection.shape[1])
        dtype = model.visual.conv1.weight.dtype
        model.segearth_upsampler = build_segearth_upsampler(
            feat_dim=feat_dim,
            device=device,
            dtype=dtype,
        )

    return model
