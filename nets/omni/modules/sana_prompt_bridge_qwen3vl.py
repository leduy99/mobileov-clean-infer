"""
Qwen3-VL prompt bridge modules for Q1 distillation.

This module keeps the current trainer shape intact by exposing the same
student-side attributes used by the Smol bridge (`smolvlm2_model`,
`projector`, `forward`, etc.), but swaps the frozen text backbone to
Qwen3-VL.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nets.omni.modules.adapter import DM_Adapter
from nets.omni.modules.sana_prompt_bridge import (
    MCPProjector,
    SanaBridgeResampler,
    build_strict_sana_select_index,
)


logger = logging.getLogger(__name__)


def _import_qwen3_vl():
    try:
        from transformers import AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Qwen3-VL dependencies. "
            "Set PYTHONPATH to include the local Qwen dependency bundle, "
            "for example: export PYTHONPATH=\"$ROOT_DIR/.tmp/qwen3deps:$ROOT_DIR:$PYTHONPATH\""
        ) from exc
    return AutoTokenizer, Qwen3VLModel


class Qwen3VLSanaPromptBridge(nn.Module):
    """
    Qwen3-VL text backbone -> bridge projector -> SANA prompt embeddings.

    This variant is intentionally text-only for the current bridge-only
    distillation experiments.
    """

    def __init__(
        self,
        qwen_ckpt_path: str,
        adapter_ckpt_dir: Optional[str],
        adapter_in_channels: int = 2048,
        adapter_out_channels: int = 2304,
        adapter_query_length: int = 128,
        adapter_num_encoder_layers: int = 2,
        adapter_num_decoder_layers: int = 2,
        adapter_ff_mult: int = 2,
        num_prompt_queries: int = 300,
        caption_channels: int = 2304,
        precision_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        tokenizer_model_id: Optional[str] = None,
        force_adapter_query_length: Optional[int] = None,
        max_length: int = 512,
        eps: float = 1e-6,
        use_vision_head: bool = False,
        resampler_num_heads: int = 8,
        resampler_mlp_mult: int = 2,
        lora_enable: bool = False,
        gate_min_value: float = 0.0,
        projector_type: str = "mcp_full",
        mcp_hidden_dim: int = 1536,
        mcp_num_fuse_layers: int = 2,
        mcp_use_refine: bool = True,
        mcp_refine_kernel_size: int = 3,
        mcp_fusion_temperature: float = 1.0,
        mcp_lexical_bottleneck_dim: int = 256,
        mcp_lexical_gate_init: float = 0.05,
        strict_sana_parity_text_path: bool = False,
        strict_sana_use_full_text_window: bool = False,
        strict_sana_token_select_strategy: str = "tail",
        strict_sana_head_tokens: int = 96,
        strict_sana_tail_tokens: int = 96,
        fail_fast_mask: Optional[bool] = None,
        sana_model_max_length: int = 300,
        sana_chi_prompt: str = "",
    ):
        super().__init__()

        if lora_enable:
            raise NotImplementedError("Qwen3-VL bridge currently supports bridge-only runs only (lora_enable=false).")
        if use_vision_head:
            raise NotImplementedError("Qwen3-VL bridge currently supports text-only mode only (use_vision_head=false).")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.caption_channels = int(caption_channels)
        self.tokenizer_model_id = str(tokenizer_model_id or qwen_ckpt_path)
        self.max_length = int(max_length)
        self.gate_min_value = float(gate_min_value)
        self.projector_type = str(projector_type or "legacy").lower()
        self.mcp_fusion_temperature = float(mcp_fusion_temperature)
        self.strict_sana_parity_text_path = bool(strict_sana_parity_text_path)
        self.strict_sana_use_full_text_window = bool(strict_sana_use_full_text_window)
        self.strict_sana_token_select_strategy = str(strict_sana_token_select_strategy or "tail").strip().lower()
        self.strict_sana_head_tokens = int(strict_sana_head_tokens)
        self.strict_sana_tail_tokens = int(strict_sana_tail_tokens)
        self.fail_fast_mask = bool(
            self.strict_sana_parity_text_path if fail_fast_mask is None else fail_fast_mask
        )
        self.sana_model_max_length = int(sana_model_max_length)
        self.sana_chi_prompt = str(sana_chi_prompt or "")
        self._chi_prompt_token_count = None
        self.use_vision_head = False
        self.smolvlm2_vision_head = None
        self._cached_tokenizer = None

        AutoTokenizer, Qwen3VLModel = _import_qwen3_vl()
        self._tokenizer_cls = AutoTokenizer
        logger.info("Loading Qwen3-VL backbone from %s", qwen_ckpt_path)
        self.qwen_model = Qwen3VLModel.from_pretrained(
            qwen_ckpt_path,
            torch_dtype=precision_dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device)
        logger.info("Loaded Qwen3-VL backbone from %s", qwen_ckpt_path)
        self.qwen_model.eval().requires_grad_(False)

        # Keep trainer/checkpoint compatibility by exposing the frozen text
        # backbone under the same attribute name the current code expects.
        self.smolvlm2_model = self.qwen_model

        qwen_hidden_size = None
        cfg = getattr(self.qwen_model, "config", None)
        if cfg is not None and hasattr(cfg, "text_config") and cfg.text_config is not None:
            qwen_hidden_size = getattr(cfg.text_config, "hidden_size", None)
        if qwen_hidden_size is None:
            qwen_hidden_size = getattr(cfg, "hidden_size", 2048)
        logger.info("Detected Qwen3-VL hidden_size=%s", qwen_hidden_size)

        if self.projector_type in ("mcp_tiny", "mcp_full"):
            if self.projector_type == "mcp_tiny":
                mcp_num_fuse_layers = max(1, min(int(mcp_num_fuse_layers), 4))
                mcp_use_refine = False
            self.projector = MCPProjector(
                d_vlm=qwen_hidden_size,
                d_cond=caption_channels,
                d_h=int(mcp_hidden_dim),
                num_fuse_layers=int(mcp_num_fuse_layers),
                use_refine=bool(mcp_use_refine),
                refine_kernel_size=int(mcp_refine_kernel_size),
            )
            self.adapter = nn.Identity()
            self.adapter_output_norm = nn.Identity()
            self.adapter_output_gate = nn.Parameter(torch.tensor([1.0], dtype=precision_dtype))
            self.adapter_output_gate.requires_grad_(False)
            self.resampler = nn.Identity()
            logger.info(
                "Using Qwen MCP projector: type=%s d_vlm=%s d_cond=%s K=%s refine=%s",
                self.projector_type,
                qwen_hidden_size,
                caption_channels,
                int(mcp_num_fuse_layers),
                bool(mcp_use_refine),
            )
        else:
            if adapter_in_channels != qwen_hidden_size:
                logger.warning(
                    "Adapter in_channels (%s) != Qwen3-VL hidden_size (%s); overriding adapter_in_channels.",
                    adapter_in_channels,
                    qwen_hidden_size,
                )
                adapter_in_channels = qwen_hidden_size
            if force_adapter_query_length is not None and force_adapter_query_length != adapter_query_length:
                logger.info(
                    "Adapter query_length forced: %s -> %s",
                    adapter_query_length,
                    force_adapter_query_length,
                )
                adapter_query_length = int(force_adapter_query_length)

            device_id = device.index if device.type == "cuda" and device.index is not None else 0
            self.adapter = DM_Adapter(
                in_channels=adapter_in_channels,
                out_channels=adapter_out_channels,
                learnable_query_length=adapter_query_length,
                num_encoder_layers=adapter_num_encoder_layers,
                num_decoder_layers=adapter_num_decoder_layers,
                ff_mult=adapter_ff_mult,
                TRAINABLE_PRECISION=precision_dtype,
                device_id=device_id,
                rank=0,
                dit_fsdp=False,
                use_usp=False,
                load_ckpt_dir=adapter_ckpt_dir,
            )

            if adapter_ckpt_dir is not None:
                self.adapter.load_ckpt()

            self.adapter_output_norm = nn.LayerNorm(adapter_out_channels, eps=eps)
            self.adapter_output_gate = nn.Parameter(torch.tensor([1e-3], dtype=precision_dtype))
            self.resampler = SanaBridgeResampler(
                in_dim=adapter_out_channels,
                out_dim=caption_channels,
                num_queries=num_prompt_queries,
                num_heads=resampler_num_heads,
                mlp_mult=resampler_mlp_mult,
                dropout=0.0,
            )
            self.projector = None

        self.to(device=device, dtype=precision_dtype)

    def _get_tokenizer(self):
        if self._cached_tokenizer is None:
            self._cached_tokenizer = self._tokenizer_cls.from_pretrained(
                self.tokenizer_model_id,
                trust_remote_code=True,
                local_files_only=True,
                padding_side="right",
            )
            if self._cached_tokenizer.pad_token is None:
                self._cached_tokenizer.pad_token = (
                    self._cached_tokenizer.eos_token
                    or self._cached_tokenizer.bos_token
                    or self._cached_tokenizer.unk_token
                )
        return self._cached_tokenizer

    def encode_prompts(
        self,
        prompts: List[str],
        return_mask: bool = False,
        return_all_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        tokenizer = self._get_tokenizer()
        prompts = [
            p if p is not None and str(p).strip() else (tokenizer.eos_token or tokenizer.pad_token or " ")
            for p in prompts
        ]
        tokenize_max_length = int(self.max_length)
        tokenize_padding = True
        if self.strict_sana_parity_text_path:
            if self.sana_model_max_length < 1:
                raise RuntimeError(f"Invalid sana_model_max_length={self.sana_model_max_length}")
            has_chi_prefix = bool(self.sana_chi_prompt) and all(str(p).startswith(self.sana_chi_prompt) for p in prompts)
            if self.strict_sana_use_full_text_window:
                tokenize_max_length = int(self.max_length)
            elif has_chi_prefix:
                if self._chi_prompt_token_count is None:
                    self._chi_prompt_token_count = len(tokenizer.encode(self.sana_chi_prompt))
                tokenize_max_length = int(self._chi_prompt_token_count + self.sana_model_max_length - 2)
                tokenize_max_length = max(tokenize_max_length, self.sana_model_max_length)
            else:
                tokenize_max_length = int(self.sana_model_max_length)
            if tokenize_max_length > int(self.max_length):
                raise RuntimeError(
                    "Strict SANA-parity requires max_length >= tokenize_max_length, "
                    f"got max_length={self.max_length}, tokenize_max_length={tokenize_max_length}"
                )
            tokenize_padding = "max_length"

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=tokenize_padding,
            truncation=True,
            max_length=tokenize_max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state
        all_hidden_states = None
        if return_all_hidden_states and hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)):
            all_hidden_states = [h for h in outputs.hidden_states if isinstance(h, torch.Tensor) and h.dim() == 3]

        if self.strict_sana_parity_text_path:
            target_len = int(self.sana_model_max_length)
            cur_len = int(hidden_states.shape[1])
            if cur_len < target_len:
                raise RuntimeError(
                    f"Strict SANA-parity requires hidden length >= {target_len}, got {cur_len}."
                )
            if cur_len != target_len:
                select_index = build_strict_sana_select_index(
                    cur_len,
                    target_len,
                    strategy=self.strict_sana_token_select_strategy,
                    head_tokens=self.strict_sana_head_tokens,
                    tail_tokens=self.strict_sana_tail_tokens,
                    device=hidden_states.device,
                )
                hidden_states = hidden_states.index_select(1, select_index)
                if attention_mask is not None:
                    attention_mask = attention_mask.index_select(1, select_index.to(attention_mask.device))
                if all_hidden_states is not None:
                    all_hidden_states = [h.index_select(1, select_index.to(h.device)) for h in all_hidden_states]

        if return_mask:
            return hidden_states, attention_mask, all_hidden_states
        return hidden_states, None, all_hidden_states

    def forward(self, prompts: List[str], return_mask: bool = False, return_aux: bool = False):
        need_all = self.projector_type in ("mcp_tiny", "mcp_full")
        hidden_states, attention_mask, all_hidden_states = self.encode_prompts(
            prompts,
            return_mask=return_mask,
            return_all_hidden_states=need_all,
        )
        aux: Dict[str, Any] = {}

        if need_all:
            if all_hidden_states is None:
                raise RuntimeError("MCP projector requires output hidden_states, got None")
            if return_aux and len(all_hidden_states) > 0:
                aux["hidden0"] = all_hidden_states[0]
            prompt_embeds = self.projector(all_hidden_states, temperature=self.mcp_fusion_temperature)
        else:
            adapter_output = self.adapter(hidden_states)
            if adapter_output.dim() == 2:
                adapter_output = adapter_output.unsqueeze(0)
            adapter_output = self.adapter_output_norm(adapter_output)
            gate = (
                torch.clamp(self.adapter_output_gate, min=self.gate_min_value)
                if self.gate_min_value > 0.0
                else self.adapter_output_gate
            )
            adapter_output = adapter_output * gate
            prompt_embeds = self.resampler(adapter_output)

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        if return_mask:
            if attention_mask is not None and attention_mask.shape[1] == prompt_embeds.shape[1]:
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                prompt_mask = attention_mask.to(device=prompt_embeds.device, dtype=torch.long)
            else:
                if self.fail_fast_mask:
                    raise RuntimeError(
                        "Fail-fast mask enabled: attention_mask/prompt_embeds shape mismatch: "
                        f"attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None}, "
                        f"prompt_embeds={tuple(prompt_embeds.shape)}"
                    )
                if not hasattr(self, "_mask_mismatch_warned"):
                    logger.warning(
                        "Prompt mask fallback to all-ones: attention_mask shape=%s, prompt_embeds shape=%s",
                        tuple(attention_mask.shape) if attention_mask is not None else None,
                        tuple(prompt_embeds.shape),
                    )
                    self._mask_mismatch_warned = True
                prompt_mask = torch.ones(
                    prompt_embeds.shape[:2],
                    device=prompt_embeds.device,
                    dtype=torch.long,
                )
            if return_aux:
                return prompt_embeds, prompt_mask, aux
            return prompt_embeds, prompt_mask
        if return_aux:
            return prompt_embeds, aux
        return prompt_embeds
