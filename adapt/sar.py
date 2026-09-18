"""SAR baseline for open-vocabulary semantic segmentation.

This adapts the official SAR algorithm (ICLR 2023) to dense segmentation:
- only visual-encoder LayerNorm affine parameters are updated, matching DAF's TENT baseline;
- reliability filtering is applied per pixel over segmentation entropy;
- SAM performs the two-step sharpness-aware update;
- model recovery uses the entropy EMA criterion from SAR.

Official implementation: https://github.com/mr-eggplant/SAR
"""

import copy
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import (
    load_prompts_from_yaml,
    print_clip_parameters,
    print_optimizer_parameters,
)

from .sam import SAM


REFERENCE_PROMPT = "a photo of a {}"


class SAR:
    """Sharpness-Aware and Reliable test-time adaptation for OVSS."""

    def __init__(
        self,
        ovss_type,
        ovss_backbone,
        lr,
        classes,
        steps=1,
        prompt_dir=None,
        runtime_calculation=False,
        reset_mode="continual",
        device="cpu",
        token_merge=False,
        merge_type="algm",
        algm_layers=(1, 7),
        algm_threshold=0.8,
        algm_window_size=(2, 2),
        sar_margin_e0=0.4,
        sar_reset_constant_em=0.2,
        sar_rho=0.05,
        sar_adaptive=False,
        sar_base_optimizer="sgd",
    ):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.classes = classes
        self.steps = steps
        self.prompt_dir = prompt_dir
        self.runtime = runtime_calculation
        self.reset_mode = reset_mode
        self.device = device
        self.token_merge = token_merge
        self.merge_type = merge_type
        self.algm_layers = algm_layers
        self.algm_threshold = algm_threshold
        self.algm_window_size = algm_window_size

        self.sar_margin_e0 = sar_margin_e0
        self.sar_reset_constant_em = sar_reset_constant_em
        self.sar_rho = sar_rho
        self.sar_adaptive = sar_adaptive
        self.sar_base_optimizer = sar_base_optimizer

        if self.classes is None or len(self.classes) == 0:
            raise ValueError("SAR requires a non-empty class list")
        if self.steps < 1:
            raise ValueError("SAR requires steps >= 1")

        # SAR uses E_0 = coefficient * log(K). 0.4 is the official coefficient.
        self.margin_e0 = float(self.sar_margin_e0) * math.log(max(len(self.classes), 2))
        self.ema = None

        self.model, self.tokenize = load_ovss(
            self.ovss_type,
            self.ovss_backbone,
            device=self.device,
            token_merge=self.token_merge,
            merge_type=self.merge_type,
            algm_layers=self.algm_layers,
            algm_threshold=self.algm_threshold,
            algm_window_size=self.algm_window_size,
        )

        if self.prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(self.prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # SAR is optimized in train mode. Keep the text tower frozen and adapt
        # only the visual normalization parameters selected by SAR.
        self.model.train()
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.configure_sar_model(self.model.visual)

        params, _ = self.collect_ln_params(self.model.visual)
        if not params:
            raise RuntimeError(
                "SAR found no visual LayerNorm parameters to adapt. "
                "Use a ViT/LN backbone or adjust the parameter collector."
            )

        base_optimizer, optimizer_kwargs = self._get_base_optimizer(
            self.sar_base_optimizer, self.lr
        )
        self.optimizer = SAM(
            params,
            base_optimizer,
            rho=self.sar_rho,
            adaptive=self.sar_adaptive,
            **optimizer_kwargs,
        )

        print_clip_parameters(self.model)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

        self.reliability_stats = []

    def adapt(self, x):
        if self.reset_mode == "episodic":
            self.reset()
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(
            x, self.text_x, True, interpolate=True
        )
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset SAR without saved model/optimizer state")
        self.load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.ema = None

    def perform_adaptation(self, x):
        """Run one or more SAR updates.

        Original SAR filters *samples* by predictive entropy before the first
        and second SAM passes. For dense segmentation, each input crop is
        treated as one sample: pixel-wise entropy is averaged spatially (and
        across prompt templates) to obtain one entropy value per crop.
        """
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            self.optimizer.zero_grad()

            logits, _, _ = self.model(
                x, self.text_x, True, interpolate=False
            )
            entropy_map = self.softmax_entropy(logits)
            entropy_per_sample = entropy_map.mean(dim=(0, -2, -1))
            reliable_mask_1 = entropy_per_sample < self.margin_e0
            reliable_1 = int(reliable_mask_1.sum().item())
            total = int(entropy_per_sample.numel())

            if reliable_1 == 0:
                loss_report.append(float(entropy_per_sample.mean().detach().item()))
                self.reliability_stats.append(
                    {"total": total, "reliable_first": 0, "reliable_second": 0}
                )
                continue

            loss_first = entropy_per_sample[reliable_mask_1].mean()
            loss_first.backward()
            self.optimizer.first_step(zero_grad=True)

            logits_second, _, _ = self.model(
                x, self.text_x, True, interpolate=False
            )
            entropy_map_second = self.softmax_entropy(logits_second)
            entropy_per_sample_second = entropy_map_second.mean(dim=(0, -2, -1))

            # Original SAR keeps only samples selected by the first pass, then
            # applies the same reliable-entropy filter again at w + epsilon.
            entropy_second_selected = entropy_per_sample_second[reliable_mask_1]
            reliable_mask_2 = entropy_second_selected < self.margin_e0
            reliable_2 = int(reliable_mask_2.sum().item())

            if reliable_2 == 0:
                # Avoid propagating NaNs when the second filtering stage is
                # empty; restore the SAM perturbation and skip the update.
                self.optimizer.restore_step(zero_grad=True)
                loss_report.append(float(loss_first.detach().item()))
                self.reliability_stats.append(
                    {
                        "total": total,
                        "reliable_first": reliable_1,
                        "reliable_second": 0,
                    }
                )
                continue

            loss_second = entropy_second_selected[reliable_mask_2].mean()
            loss_second_value = float(loss_second.detach().item())
            loss_second.backward()
            self.optimizer.second_step(zero_grad=True)

            self.ema = self.update_ema(self.ema, loss_second_value)
            loss_report.append(loss_second_value)
            self.reliability_stats.append(
                {
                    "total": total,
                    "reliable_first": reliable_1,
                    "reliable_second": reliable_2,
                }
            )

            # Model recovery from the original SAR algorithm.
            if (
                self.ema is not None
                and self.sar_reset_constant_em >= 0
                and self.ema < self.sar_reset_constant_em
            ):
                self.reset()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)

        return loss_report

    @staticmethod
    def update_ema(ema, new_data):
        if ema is None:
            return new_data
        return 0.9 * ema + 0.1 * new_data

    @staticmethod
    def softmax_entropy(logits):
        """Pixel-wise entropy for [templates, batch, classes, H, W] logits."""
        return -(logits.softmax(-3) * logits.log_softmax(-3)).sum(-3)

    @staticmethod
    def _sar_skip_ln(name):
        """Mirror the official SAR ViT-Base parameter selection on CLIP names."""
        if name == "ln_post":
            return True
        # Official SAR excludes ViT-Base blocks 9--11 and the final norm.
        for idx in (9, 10, 11):
            if name.startswith(f"transformer.resblocks.{idx}."):
                return True
        return False

    @classmethod
    def configure_sar_model(cls, model):
        model.train()
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm) and not cls._sar_skip_ln(name):
                module.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        params = []
        names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name in ("weight", "bias") and param.requires_grad:
                        params.append(param)
                        names.append(f"visual.{name}.{param_name}")
        return params, names

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [prompt.format(class_name) for prompt in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            if average:
                avg_embedding = class_embeddings.mean(dim=0)
                avg_embedding = avg_embedding / avg_embedding.norm()
                class_embeddings = torch.cat(
                    [class_embeddings, avg_embedding.unsqueeze(0)], dim=0
                )
            text_features.append(class_embeddings)
        return torch.stack(text_features, dim=1).to(self.device)

    @staticmethod
    def _get_base_optimizer(name, lr):
        name = name.lower()
        if name == "sgd":
            return optim.SGD, dict(lr=lr, momentum=0.9, weight_decay=0.0)
        if name == "adam":
            return optim.Adam, dict(
                lr=lr, betas=(0.9, 0.999), weight_decay=0.0
            )
        if name == "adamw":
            return optim.AdamW, dict(
                lr=lr, betas=(0.9, 0.999), weight_decay=0.0
            )
        raise ValueError(f"Unsupported SAR base optimizer: {name}")

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
