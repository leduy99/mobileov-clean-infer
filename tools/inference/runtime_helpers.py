from __future__ import annotations

import random
from dataclasses import is_dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


def to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    if is_dataclass(obj):
        return to_attrdict(obj.__dict__)
    return obj


def normalize_prompt(text: str, normalize_whitespace: bool, strip: bool, remove_double_newlines: bool) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n")
    if remove_double_newlines:
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
    text = text.replace("<image>", "").replace("<video>", "")
    if normalize_whitespace:
        text = " ".join(text.split())
    if strip:
        text = text.strip()
    return text


def truncate_prompt(prompt: str, tokenizer, max_tokens: Optional[int]) -> str:
    if not prompt or max_tokens is None:
        return prompt
    try:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception:
        return prompt
    if len(token_ids) <= max_tokens:
        return prompt
    truncated = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
    return truncated.strip()


def _apply_template(prompt: str, templates: List[str], motion_score: int, rng: random.Random) -> str:
    if not templates:
        return prompt
    template = rng.choice(templates)
    if "{motion_score}" in template:
        return template.format(prompt=prompt, motion_score=int(motion_score))
    return template.format(prompt=prompt)


def preprocess_prompts(
    prompts: List[str],
    cfg: AttrDict,
    rng: random.Random,
    tokenizer=None,
    chi_prompt: Optional[str] = None,
) -> List[str]:
    prep_cfg = cfg.data.get("preprocessing", AttrDict())
    normalize_whitespace = bool(getattr(prep_cfg, "normalize_whitespace", True))
    strip = bool(getattr(prep_cfg, "strip", True))
    remove_double_newlines = bool(getattr(prep_cfg, "remove_double_newlines", True))
    use_chi_prompt = bool(getattr(prep_cfg, "use_chi_prompt", False))
    use_prompt_templates_cfg = getattr(prep_cfg, "use_prompt_templates", None)
    use_prompt_templates = bool(use_chi_prompt if use_prompt_templates_cfg is None else use_prompt_templates_cfg)
    max_prompt_tokens = getattr(prep_cfg, "max_prompt_tokens", None)

    templates = cfg.data.get("prompt_templates", []) if use_prompt_templates else []
    motion_score = int(cfg.data.get("motion_score", 10))

    processed = []
    for prompt in prompts:
        text = normalize_prompt(prompt, normalize_whitespace, strip, remove_double_newlines)
        if max_prompt_tokens and tokenizer is not None:
            text = truncate_prompt(text, tokenizer, max_prompt_tokens)
        if templates:
            text = _apply_template(text, templates, motion_score, rng)
        if use_chi_prompt and chi_prompt:
            text = f"{chi_prompt}{text}"
        processed.append(text)
    return processed


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        self.r = int(max(0, r))
        self.alpha = int(max(1, alpha))
        self.scaling = float(self.alpha) / float(self.r) if self.r > 0 else 1.0
        if self.r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros(self.r, self.in_features, device=base.weight.device, dtype=base.weight.dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_input = F.dropout(x, p=self.dropout, training=self.training)
            update = F.linear(lora_input, self.lora_A)
            update = F.linear(update, self.lora_B)
            result = result + update * self.scaling
        return result


def apply_lora_to_module(
    module: torch.nn.Module,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> int:
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    target_set = set(target_modules or [])

    named_modules = list(module.named_modules())
    replaced = 0
    for name, submodule in named_modules:
        if not isinstance(submodule, nn.Linear):
            continue
        if include_patterns and not any(pat in name for pat in include_patterns):
            continue
        if exclude_patterns and any(pat in name for pat in exclude_patterns):
            continue
        if target_set and not any(name.endswith(t) or t in name for t in target_set):
            continue

        parent_path = name.split(".")[:-1]
        leaf = name.split(".")[-1]
        parent = module
        for p in parent_path:
            parent = getattr(parent, p)
        wrapped = LoRALinear(submodule, r=r, alpha=alpha, dropout=dropout)
        wrapped.to(submodule.weight.device, dtype=submodule.weight.dtype)
        if hasattr(parent, "_modules") and leaf in parent._modules:
            parent._modules[leaf] = wrapped
            replaced += 1

    for param in module.parameters():
        param.requires_grad = False
    for submodule in module.modules():
        if hasattr(submodule, "lora_A"):
            submodule.lora_A.requires_grad = True
        if hasattr(submodule, "lora_B"):
            submodule.lora_B.requires_grad = True
    return replaced
