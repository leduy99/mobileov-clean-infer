"""
Mobile-OV Model: Unified architecture with SmolVLM2 for understanding and WAN for generation.

This model replaces the understanding module (VisionHead) with SmolVLM2-500M,
while keeping the generation module (WAN) unchanged.

Future plans:
- Experiment 2: Replace generation module with SANA-video
- Experiment 3: Full unified model with SmolVLM2 + SANA
"""

import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import einops
from typing import List, Dict, Optional, Union, Tuple
import random

from nets.third_party.wan.modules.model import WanModel
from nets.omni.modules.adapter import DM_Adapter
from nets.omni.modules.visual_context_adapter import VisualContextAdapter
from nets.omni.modules.smolvlm2_vision_head import SmolVLM2VisionHead
from nets.smolvlm2 import load_smolvlm2_from_ckpt, SmolVLMModel, SmolVLMForConditionalGeneration

logger = logging.getLogger(__name__)


class MobileOVModel(nn.Module):
    """
    Mobile-OV Model: SmolVLM2 (understanding) + WAN (generation)
    
    This model integrates SmolVLM2-500M as the understanding module,
    replacing the original VisionHead. The generation module (WAN) remains unchanged.
    
    Key differences from OmniVideoMixedConditionModel:
    - Uses SmolVLM2 instead of VisionHead for understanding
    - Encodes prompts on-the-fly during forward pass (no pre-computed features needed)
    - Can optionally use pre-computed features for backward compatibility
    """
    
    def __init__(
        self,
        wan_model_or_ckpt_dir: Union[WanModel, str],
        adapter_or_ckpt_dir: Union[DM_Adapter, str],
        smolvlm2_ckpt_path: Optional[str] = None,
        smolvlm2_model: Optional[SmolVLMModel] = None,
        adapter_in_channels: int = 1152,
        adapter_out_channels: int = 4096,
        adapter_query_length: int = 64,
        precision_dtype: torch.dtype = torch.float32,
        device_id: int = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        use_visual_context_adapter: bool = False,
        visual_context_adapter_patch_size: tuple = None,
        max_context_len = None,
        eps: float = 1e-6,
        use_precomputed_features: bool = False,  # For backward compatibility
        disable_t5_context: bool = False,  # If True, skip T5 context when using SmolVLM2 (avoid double-conditioning)
        use_smol_vh: bool = True,  # Use VisionHead-style resampler (recommended, matches OmniVideo)
        smol_vh_num_queries: int = 1,  # Number of query tokens: 1 for T2V bring-up, 4 for richer clues
    ):
        """
        Initialize Mobile-OV model with SmolVLM2 understanding module.
        
        Args:
            wan_model_or_ckpt_dir: Either a WanModel instance or a path to a checkpoint directory
            adapter_or_ckpt_dir: Either a DM_Adapter instance or a path to a checkpoint directory
            smolvlm2_ckpt_path: Path to converted SmolVLM2 checkpoint (.pt file)
            smolvlm2_model: Optional pre-loaded SmolVLM2 model instance
            adapter_in_channels: Input channels for adapter
            adapter_out_channels: Output channels for adapter
            adapter_query_length: Query length for adapter
            precision_dtype: Precision type for computation
            device_id: GPU device ID
            rank: Process rank
            dit_fsdp: Whether to use FSDP for DiT model
            use_usp: Whether to use USP
            use_visual_context_adapter: Whether to use visual context adapter
            visual_context_adapter_patch_size: Patch size for visual context adapter
            max_context_len: Maximum context length
            eps: Epsilon for numerical stability
            use_precomputed_features: If True, use pre-computed features from dataset (backward compat)
            disable_t5_context: If True, skip T5 context when using SmolVLM2 to avoid double-conditioning
        """
        super().__init__()
        self.disable_t5_context = disable_t5_context
        self.use_smol_vh = use_smol_vh
        self.smol_vh_num_queries = smol_vh_num_queries
        
        # FIX: Add LayerNorm + learnable gate for adapter output alignment
        # This helps align adapter tokens to T5 distribution space
        # LayerNorm normalizes the adapter output
        # Learnable gate (initialized small) allows gradual integration
        self.adapter_output_norm = nn.LayerNorm(adapter_out_channels, eps=eps)
        # Initialize gate to small value (1e-3) to start with minimal clue influence
        self.adapter_output_gate = nn.Parameter(torch.tensor(1e-3, dtype=precision_dtype))
        
        # Handle WanModel initialization
        if isinstance(wan_model_or_ckpt_dir, str):
            self.wan_model = WanModel.from_pretrained(wan_model_or_ckpt_dir)
        else:
            self.wan_model = wan_model_or_ckpt_dir

        if max_context_len is not None:
            self.wan_model.text_len = max_context_len
        self.max_context_len = self.wan_model.text_len
        
        # Handle DM_Adapter initialization
        self.adapter = DM_Adapter(
            in_channels=adapter_in_channels,
            out_channels=adapter_out_channels,
            learnable_query_length=adapter_query_length,
            TRAINABLE_PRECISION=precision_dtype,
            device_id=device_id,
            rank=rank,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            load_ckpt_dir=adapter_or_ckpt_dir
        )
        if adapter_or_ckpt_dir is not None and os.path.exists(adapter_or_ckpt_dir):
            logger.info(f"Loading adapter from {adapter_or_ckpt_dir}")
            # Check if it's a directory or file
            if os.path.isdir(adapter_or_ckpt_dir):
                # Use load_ckpt() which looks for adapter_pytorch_model.bin in the directory
                self.adapter.load_ckpt()
            else:
                # It's a file, use load_checkpoint()
                self.adapter.load_checkpoint(adapter_or_ckpt_dir)
        
        # Initialize SmolVLM2 understanding module
        self.use_precomputed_features = use_precomputed_features
        if use_precomputed_features:
            logger.info("Using pre-computed features mode (backward compatibility)")
            self.smolvlm2_model = None
        else:
            if smolvlm2_model is not None:
                self.smolvlm2_model = smolvlm2_model
            elif smolvlm2_ckpt_path is not None and os.path.exists(smolvlm2_ckpt_path):
                logger.info(f"Loading SmolVLM2 from {smolvlm2_ckpt_path}")
                device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
                
                # Check if checkpoint has lm_head to determine model class
                try:
                    checkpoint = torch.load(smolvlm2_ckpt_path, map_location="cpu", weights_only=False)
                    has_lm_head = False
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        has_lm_head = any('lm_head' in k for k in checkpoint["state_dict"].keys())
                    
                    if has_lm_head:
                        logger.info("Checkpoint has lm_head, loading SmolVLMForConditionalGeneration...")
                        self.smolvlm2_model = load_smolvlm2_from_ckpt(
                            smolvlm2_ckpt_path, 
                            device=device,
                            model_class=SmolVLMForConditionalGeneration
                        )
                    else:
                        logger.info("Checkpoint does not have lm_head, loading base SmolVLMModel...")
                        self.smolvlm2_model = load_smolvlm2_from_ckpt(smolvlm2_ckpt_path, device=device)
                except Exception as e:
                    logger.warning(f"Could not check checkpoint for lm_head: {e}, using default loader...")
                    self.smolvlm2_model = load_smolvlm2_from_ckpt(smolvlm2_ckpt_path, device=device)
                
                self.smolvlm2_model.eval()  # Default to eval mode
            else:
                raise ValueError(
                    "Either smolvlm2_ckpt_path or smolvlm2_model must be provided, "
                    "or set use_precomputed_features=True"
                )
        
        # Projection layer to map SmolVLM2 hidden size to adapter input size
        # SmolVLM2-500M typically has hidden_size=1024, need to project to adapter_in_channels (1152)
        
        # CRITICAL FIX: Vision head should be created even when use_precomputed_features=True
        # because it's still needed to process precomputed features in training
        # The difference is:
        # - use_precomputed_features=False: SmolVLM2 encodes prompts on-the-fly, vision head processes output
        # - use_precomputed_features=True: Features are precomputed, but vision head still processes them
        smolvlm2_hidden_size = None
        
        if not use_precomputed_features and self.smolvlm2_model is not None:
            # Try to get hidden size from model config
            if hasattr(self.smolvlm2_model, 'config') and hasattr(self.smolvlm2_model.config, 'hidden_size'):
                smolvlm2_hidden_size = self.smolvlm2_model.config.hidden_size
            elif hasattr(self.smolvlm2_model, 'hidden_size'):
                smolvlm2_hidden_size = self.smolvlm2_model.hidden_size
            elif hasattr(self.smolvlm2_model, 'config') and hasattr(self.smolvlm2_model.config, 'text_config'):
                smolvlm2_hidden_size = self.smolvlm2_model.config.text_config.hidden_size
        
        # Default to 1024 if cannot detect (SmolVLM2-500M default)
        if smolvlm2_hidden_size is None:
            smolvlm2_hidden_size = 1024
            if not use_precomputed_features:
                logger.warning(f"Could not detect SmolVLM2 hidden_size, using default: {smolvlm2_hidden_size}")
        
        # Use VisionHead-style resampler (recommended, matches OmniVideo architecture)
        # CRITICAL: Create vision head even when use_precomputed_features=True
        if self.use_smol_vh:
            self.smolvlm2_vision_head = SmolVLM2VisionHead(
                llm_hidden_size=smolvlm2_hidden_size,  # 1024
                hidden_size=adapter_in_channels,       # 1152
                learnable_query_length=self.smol_vh_num_queries,  # 1 for T2V bring-up
                TRAINABLE_PRECISION=precision_dtype,
            )
            logger.info(f"Created SmolVLM2VisionHead: [B, L, {smolvlm2_hidden_size}] -> [B, {self.smol_vh_num_queries}, {adapter_in_channels}]")
            
            # VisionHead already includes fc: 1024->1152, so no need for separate projection
            self.smolvlm2_projection = None
            self.smolvlm2_resampler = None  # Legacy Resampler4, no longer used
            
            # Try to load VisionHead from checkpoint if available
            vh_loaded = False
            if smolvlm2_ckpt_path is not None:
                vh_ckpt_path = os.path.join(
                    os.path.dirname(smolvlm2_ckpt_path),
                    "smolvlm2_vision_head",
                    "pytorch_model.bin"
                )
                if os.path.exists(vh_ckpt_path):
                    logger.info(f"Loading SmolVLM2VisionHead from {vh_ckpt_path}")
                    vh_state_dict = torch.load(vh_ckpt_path, map_location="cpu")
                    self.smolvlm2_vision_head.load_state_dict(vh_state_dict)
                    logger.info("SmolVLM2VisionHead loaded successfully")
                    vh_loaded = True
            
            # Try trained components directory
            if not vh_loaded:
                for trained_dir in [
                    "output/trained_components_fixed", 
                    "output/trained_components",
                    "output/inference/trained_components",
                    "./output/inference/trained_components"
                ]:
                    trained_vh_path = os.path.join(trained_dir, "smolvlm2_vision_head", "pytorch_model.bin")
                    if os.path.exists(trained_vh_path):
                        logger.info(f"Loading trained SmolVLM2VisionHead from {trained_vh_path}")
                        vh_state_dict = torch.load(trained_vh_path, map_location="cpu")
                        self.smolvlm2_vision_head.load_state_dict(vh_state_dict)
                        logger.info("Trained SmolVLM2VisionHead loaded successfully")
                        vh_loaded = True
                        break
            
            if not vh_loaded:
                logger.info("SmolVLM2VisionHead initialized randomly (will be trained)")
        elif not use_precomputed_features and self.smolvlm2_model is not None:
            # Fallback: use projection-only path (legacy, not recommended)
            # Only create projection if we have SmolVLM2 model
            self.smolvlm2_vision_head = None
            self.smolvlm2_projection = nn.Linear(
                smolvlm2_hidden_size,
                adapter_in_channels,
                bias=False
            )
            logger.info(f"Created SmolVLM2 projection (legacy): {smolvlm2_hidden_size} -> {adapter_in_channels}")
            self.smolvlm2_resampler = None
        else:
            # use_precomputed_features=True and use_smol_vh=False: no vision head or projection needed
            self.smolvlm2_projection = None
            self.smolvlm2_resampler = None
            self.smolvlm2_vision_head = None

        # Visual Context Adapter
        if use_visual_context_adapter:
            self.visual_context_adapter = VisualContextAdapter(
                patch_size=self.wan_model.patch_size if visual_context_adapter_patch_size is None else visual_context_adapter_patch_size,
                in_channels=self.wan_model.in_dim,
                hidden_dim=self.wan_model.dim,
                out_dim=self.wan_model.text_dim,
                eps=eps,
            )
        else:
            self.visual_context_adapter = None
    
    @classmethod
    def from_pretrained(
        cls,
        wan_ckpt_dir: str,
        adapter_ckpt_dir: str,
        smolvlm2_ckpt_path: str,
        adapter_in_channels: int = 1152,
        adapter_out_channels: int = 4096,
        adapter_query_length: int = 64,
        precision_dtype: torch.dtype = torch.float32,
        device_id: int = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        use_visual_context_adapter: bool = False,
        visual_context_adapter_patch_size: tuple = None,
        max_context_len = None,
        eps: float = 1e-6,
        use_precomputed_features: bool = False,
        disable_t5_context: bool = True,  # Default True: disable T5 when using SmolVLM2 to avoid double-conditioning
        use_smol_vh: bool = True,  # Use VisionHead-style resampler (recommended)
        smol_vh_num_queries: int = 1,  # Number of query tokens: 1 for T2V bring-up
    ):
        """
        Create MobileOVModel from pretrained checkpoints.
        
        Args:
            wan_ckpt_dir: Directory containing WanModel checkpoint
            adapter_ckpt_dir: Directory containing DM_Adapter checkpoint
            smolvlm2_ckpt_path: Path to converted SmolVLM2 checkpoint (.pt file)
            ... (other args same as __init__)
            disable_t5_context: If True, skip T5 context when using SmolVLM2 (default True to avoid double-conditioning)
            
        Returns:
            MobileOVModel: The initialized model
        """
        return cls(
            wan_model_or_ckpt_dir=wan_ckpt_dir,
            adapter_or_ckpt_dir=adapter_ckpt_dir,
            smolvlm2_ckpt_path=smolvlm2_ckpt_path,
            adapter_in_channels=adapter_in_channels,
            adapter_out_channels=adapter_out_channels,
            adapter_query_length=adapter_query_length,
            precision_dtype=precision_dtype,
            device_id=device_id,
            rank=rank,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            use_visual_context_adapter=use_visual_context_adapter,
            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
            max_context_len=max_context_len,
            eps=eps,
            use_precomputed_features=use_precomputed_features,
            disable_t5_context=disable_t5_context,
            use_smol_vh=use_smol_vh,
            smol_vh_num_queries=smol_vh_num_queries,
        )
    
    def encode_prompts_with_smolvlm2(
        self,
        prompts: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode prompts using SmolVLM2 model.
        
        Args:
            prompts: List of text prompts
            device: Device to run encoding on
            
        Returns:
            Tensor of shape [B, L, D] where D is SmolVLM2 hidden size
        """
        if self.smolvlm2_model is None:
            raise RuntimeError("SmolVLM2 model not initialized")
        
        # Get tokenizer (with caching)
        tokenizer = self.smolvlm2_model.get_tokenizer()
        if tokenizer is None:
            # Fallback: Try to load tokenizer from HuggingFace (cache it)
            if not hasattr(self, '_cached_tokenizer'):
                logger.warning("Tokenizer not found in checkpoint, trying to load from HuggingFace...")
                try:
                    from transformers import AutoTokenizer
                    # SmolVLM2 model name on HuggingFace
                    self._cached_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
                    logger.info("✓ Loaded tokenizer from HuggingFace (cached)")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer from HuggingFace: {e}")
                    raise RuntimeError("Tokenizer not available in SmolVLM2 model and HuggingFace fallback failed")
            tokenizer = self._cached_tokenizer
        
        # Tokenize prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Forward through SmolVLM2
        # CRITICAL FIX: When SmolVLM2 is frozen (requires_grad=False), PyTorch automatically
        # skips gradient computation even without torch.no_grad(). To allow gradients to flow
        # through to downstream modules (vision_head, adapter), we need to ensure at least
        # one parameter has requires_grad=True OR use a detach+requires_grad trick.
        # 
        # However, the real issue is: if ALL SmolVLM2 parameters have requires_grad=False,
        # the output will not have gradients. We've set requires_grad=True in finetune_model.py,
        # but we need to ensure the forward pass actually creates gradients.
        
        # Forward pass - gradients will flow if SmolVLM2 has any requires_grad=True params
        outputs = self.smolvlm2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract last hidden state (handle both SmolVLMModel and SmolVLMForConditionalGeneration outputs)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state  # [B, L, D]
        elif hasattr(outputs, "hidden_states") and isinstance(outputs.hidden_states, (list, tuple)) and len(outputs.hidden_states) > 0:
            # For SmolVLMForConditionalGeneration, hidden_states is a tuple
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            # Fallback: if output is tuple, take first element
            hidden_states = outputs[0]
        else:
            # Debug: print output structure
            logger.error(f"SmolVLM2 output type: {type(outputs)}")
            logger.error(f"SmolVLM2 output attributes: {dir(outputs)}")
            if hasattr(outputs, "hidden_states"):
                logger.error(f"hidden_states type: {type(outputs.hidden_states)}")
            raise RuntimeError(f"SmolVLM2 output does not have last_hidden_state. Output type: {type(outputs)}, attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
        
        # Apply attention mask to zero-out padding tokens
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        
        # CRITICAL: If hidden_states doesn't have gradients (SmolVLM2 fully frozen),
        # we need to create a gradient path. The trick: detach and require_grad creates
        # a new leaf tensor that can receive gradients from downstream.
        # But this breaks the computation graph! Instead, we should ensure SmolVLM2
        # parameters have requires_grad=True (which we did in finetune_model.py).
        # If it still doesn't work, the issue might be elsewhere in the forward pass.
        
        return hidden_states
    
    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor] = None,
        aligned_emb: Optional[torch.Tensor] = None,
        ar_vision_input: Optional[List[torch.Tensor]] = None,
        prompts: Optional[List[str]] = None,  # NEW: prompts for SmolVLM2 encoding
        visual_emb: Optional[torch.Tensor] = None,
        ref_images: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        special_token_dict: Optional[Dict[str, torch.Tensor]] = None,
        classifier_free_ratio: Optional[float] = None,
        unconditioned_context: Optional[dict[torch.Tensor]] = None,
        condition_mode: str = "full",
        gamma: float = 1.0,  # DEPRECATED: Always use full clue (gamma=1.0). Kept for backward compatibility.
        precomputed_adapter_output: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,  # OPTIMIZATION: Pre-computed adapter output to skip adapter computation
        return_adapter_output: bool = False,  # NEW: If True, return adapter_output for distillation loss
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with SmolVLM2 understanding module.
        
        Args:
            x: Input tensor for WanModel (List[Tensor]): List of input video tensors
            t: Timestep tensor
            context: List of context tensors (text embeddings)
            aligned_emb: Aligned embeddings for the adapter
            ar_vision_input: Pre-computed AR vision input (used if use_precomputed_features=True)
            prompts: Text prompts for SmolVLM2 encoding (used if use_precomputed_features=False)
            visual_emb: Visual embeddings
            ref_images: Reference images
            seq_len: Sequence length for WanModel
            special_token_dict: Dictionary of special tokens
            classifier_free_ratio: Ratio for classifier-free guidance
            unconditioned_context: Unconditioned context for classifier-free guidance
            condition_mode: Mode for conditioning
            
        Returns:
            Output from WanModel
        """
        if condition_mode == "full":
            assert x is not None, "Full mode requires at least x"
            batch_size = x.size(0) if isinstance(x, torch.Tensor) else len(x)
            # DEBUG: Print to verify forward is called
            if not hasattr(self, '_forward_called'):
                print(f"[PROFILE MobileOV] Forward called! batch_size={batch_size}", flush=True)
                self._forward_called = True
            
            # Determine which samples should use unconditional context (CFG)
            # This needs to be done BEFORE processing to ensure CFG is applied correctly
            use_uncond_mask = [False] * batch_size
            if classifier_free_ratio is not None and classifier_free_ratio > 0:
                if unconditioned_context is None:
                    raise ValueError("unconditioned_context must be provided when classifier_free_ratio > 0")
                for idx in range(batch_size):
                    if random.random() < classifier_free_ratio:
                        use_uncond_mask[idx] = True
            
            # Apply classifier-free guidance for precomputed features mode
            if self.use_precomputed_features:
                for idx in range(batch_size):
                    if use_uncond_mask[idx] and isinstance(unconditioned_context, dict):
                        if ar_vision_input is not None and ar_vision_input[idx] is not None and 'uncond_ar_vision' in unconditioned_context:
                            # CRITICAL: Clone and detach to prevent gradient flow
                            ar_vision_input[idx] = unconditioned_context['uncond_ar_vision'].clone().detach()
                        if context is not None and context[idx] is not None and 'uncond_context' in unconditioned_context:
                            # CRITICAL: Clone and detach to prevent gradient flow
                            if unconditioned_context['uncond_context'] is not None:
                                context[idx] = unconditioned_context['uncond_context'].clone().detach()
            
            # Process understanding module (SmolVLM2 or pre-computed features)
            ar_vision_head_output = None
            adapter_output = None
            
            # OPTIMIZATION: If precomputed_adapter_output is provided, use it directly (skip adapter computation)
            # This is the fastest path: adapter output already computed before denoising loop
            if precomputed_adapter_output is not None:
                # Pre-computed adapter output: use directly, skip all processing
                logger.debug(f"🚀 Using pre-computed adapter output (type: {type(precomputed_adapter_output)}) - SKIPPING adapter computation!")
                if isinstance(precomputed_adapter_output, list):
                    adapter_output = precomputed_adapter_output
                elif isinstance(precomputed_adapter_output, torch.Tensor):
                    # Convert tensor to list format expected by downstream code
                    if precomputed_adapter_output.dim() == 3:
                        adapter_output = [precomputed_adapter_output[i] for i in range(batch_size)]
                    elif precomputed_adapter_output.dim() == 2:
                        adapter_output = [precomputed_adapter_output]
                    else:
                        adapter_output = [precomputed_adapter_output]
                else:
                    adapter_output = [precomputed_adapter_output]
            # OPTIMIZATION: If ar_vision_input is provided (pre-computed), use it even if use_precomputed_features=False
            # This allows inference to pre-encode prompts once and reuse in all denoising steps
            # Similar to OmniVideo which uses pre-computed ar_vision_input
            # BUT: During training with use_precomputed_features=False, we want to train VisionHead,
            # so we should NOT use ar_vision_input from vlm_last_hidden_states (which has wrong shape [4096]).
            # Only use ar_vision_input if it's already processed (aligned_emb shape [1152])
            elif self.use_precomputed_features or (ar_vision_input is not None and self.use_precomputed_features):
                # Backward compatibility: use pre-computed features
                if ar_vision_input is not None:
                    # In this mode, ar_vision_input should already be in the right format
                    # We still need to project through adapter
                    # CRITICAL: Pre-computed features may not have gradients, but adapter
                    # parameters have requires_grad=True, so adapter output WILL have gradients
                    if isinstance(ar_vision_input, list):
                        ar_vision_head_output = [None] * batch_size
                        adapter_output = [None] * batch_size
                        for idx in range(batch_size):
                            if ar_vision_input[idx] is not None:
                                # Assume ar_vision_input is already processed (like from VisionHead)
                                # Shape should be [1, L, 1152] or similar
                                # CRITICAL: Even if ar_vision_input doesn't have gradients,
                                # adapter output will have gradients because adapter parameters do
                                ar_vision_head_output[idx] = ar_vision_input[idx]
                                adapter_output[idx] = self.adapter(ar_vision_head_output[idx])  # [1, L, 1152] -> [1, 256, 4096]
                    else:
                        # CRITICAL: Even if ar_vision_input doesn't have gradients,
                        # adapter output will have gradients because adapter parameters do
                        ar_vision_head_output = ar_vision_input
                        adapter_output = self.adapter(ar_vision_head_output)  # [B, L, 1152] -> [B, 256, 4096]
            else:
                # Use SmolVLM2 to encode prompts on-the-fly
                if prompts is None:
                    raise ValueError("prompts must be provided when use_precomputed_features=False")
                
                device = x[0].device if isinstance(x, list) else x.device
                
                # CRITICAL FIX: Replace prompts with null prompt for CFG samples
                # This ensures CFG is applied correctly when using SmolVLM2 on-the-fly
                # Use minimal prompt instead of empty string to avoid tokenization issues
                modified_prompts = []
                null_prompt_text = "a generic scene"  # Minimal prompt for unconditional generation
                for idx in range(batch_size):
                    if use_uncond_mask[idx]:
                        modified_prompts.append(null_prompt_text)  # Null prompt for unconditional generation
                    else:
                        if isinstance(prompts, list):
                            modified_prompts.append(prompts[idx])
                        else:
                            modified_prompts.append(prompts)
                
                # Encode prompts with SmolVLM2 (including null prompts for CFG samples)
                smolvlm2_hidden = self.encode_prompts_with_smolvlm2(modified_prompts, device)  # [B, L, 1024]
                
                # OmniVideo-style pipeline: SmolVLM2 -> VisionHead -> Adapter
                adapter_output = [None] * batch_size
                
                if self.use_smol_vh and self.smolvlm2_vision_head is not None:
                    # VisionHead-style resampler (matches OmniVideo architecture)
                    # VisionHead includes fc: 1024->1152 internally, no separate projection needed
                    vision_tokens = self.smolvlm2_vision_head(smolvlm2_hidden)  # [B, Q, 1152], Q=smol_vh_num_queries
                    
                    # CRITICAL: Process all samples at once to preserve gradients
                    # Pass entire batch to adapter instead of processing one by one
                    # This ensures gradients flow through the entire batch
                    adapter_output_batch = self.adapter(vision_tokens)  # [B, Q, 1152] -> [B, K, 4096] where K=64 (adapter_query_length)
                    
                    # Store adapter_output_batch for distillation loss if needed
                    if return_adapter_output:
                        self._last_adapter_output_batch = adapter_output_batch
                    
                    # CRITICAL: Keep as batch tensor, don't split into list to preserve gradients
                    # We'll handle indexing later when needed, but keep the batch tensor for now
                    # Convert to list format expected by downstream code, but preserve gradients
                    if adapter_output_batch.dim() == 3:
                        # [B, K, 4096] -> list of [K, 4096] where K=64 (adapter_query_length)
                        adapter_output = [adapter_output_batch[i] for i in range(batch_size)]
                    elif adapter_output_batch.dim() == 2:
                        # [K, 4096] -> single item list where K=64
                        adapter_output = [adapter_output_batch]
                    else:
                        adapter_output = [adapter_output_batch]
                else:
                    # Fallback: projection-only path (legacy, not recommended)
                    if self.smolvlm2_projection is not None:
                        proj_weight_dtype = next(self.smolvlm2_projection.parameters()).dtype
                        if smolvlm2_hidden.dtype != proj_weight_dtype:
                            smolvlm2_hidden = smolvlm2_hidden.to(dtype=proj_weight_dtype)
                        projected_hidden = self.smolvlm2_projection(smolvlm2_hidden)  # [B, L, 1152]
                    else:
                        logger.warning("Neither VisionHead nor projection available, using raw hidden states")
                        projected_hidden = smolvlm2_hidden  # [B, L, 1024], not ideal
                    
                    for idx in range(batch_size):
                        sample_hidden = projected_hidden[idx:idx+1]  # [1, L, C]
                        adapter_output[idx] = self.adapter(sample_hidden)  # [1, 256, 4096]
                
                # Store unconditioned adapter output for later use (don't replace adapter_output yet)
                # We'll use it when creating mixed_context to avoid grad_fn issues
                stored_uncond_adapter_output = None
                if isinstance(unconditioned_context, dict) and 'uncond_ar_vision' in unconditioned_context:
                    stored_uncond_adapter_output = unconditioned_context['uncond_ar_vision']
            
            # Process visual embeddings and reference images (same as original)
            processed_visual_emb = None if visual_emb is None else [None] * batch_size
            processed_ref_images = None if ref_images is None else [None] * batch_size
            
            if visual_emb is not None and self.visual_context_adapter is not None:
                if isinstance(visual_emb, list):
                    for idx in range(batch_size):
                        if visual_emb[idx] is not None:
                            processed_visual_emb[idx] = self.visual_context_adapter(visual_emb[idx]).squeeze(0)
                        else:
                            processed_visual_emb[idx] = None
                else:
                    processed_visual_emb = self.visual_context_adapter(visual_emb)
            
            if ref_images is not None and self.visual_context_adapter is not None:
                if isinstance(ref_images, list):
                    for idx in range(batch_size):
                        if ref_images[idx] is not None:
                            processed_ref_images[idx] = self.visual_context_adapter(ref_images[idx]).squeeze(0)
                        else:
                            processed_ref_images[idx] = None
                else:
                    processed_ref_images = self.visual_context_adapter(ref_images)
            
            # Create mixed context (same as original)
            mixed_context = []
            for idx in range(batch_size):
                components = []
                
                # Get adapter output
                # CRITICAL: Use unconditioned context if CFG is enabled for this sample
                # Otherwise use the regular adapter output
                adapter_item = None
                if use_uncond_mask[idx] and stored_uncond_adapter_output is not None:
                    # Use unconditioned context for CFG sample
                    # CRITICAL: Truncate to 64 tokens to match OmniVideo (same as conditioned path)
                    TARGET_ADAPTER_TOKENS = 64  # Match OmniVideo's adapter output length
                    
                    try:
                        if stored_uncond_adapter_output.dim() == 3:
                            # [1, K, 4096] -> [K, 4096] (truncate to 64)
                            uncond_item = stored_uncond_adapter_output[0]
                            if uncond_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                uncond_item = uncond_item[:TARGET_ADAPTER_TOKENS, :]
                        elif stored_uncond_adapter_output.dim() == 2:
                            # Already [K, 4096] (truncate to 64)
                            uncond_item = stored_uncond_adapter_output
                            if uncond_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                uncond_item = uncond_item[:TARGET_ADAPTER_TOKENS, :]
                        else:
                            # Fallback: flatten to 2D
                            uncond_item = stored_uncond_adapter_output.view(-1, stored_uncond_adapter_output.shape[-1])
                            if uncond_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                uncond_item = uncond_item[:TARGET_ADAPTER_TOKENS, :]
                        
                        # CRITICAL: Create a completely new tensor (not a view or reference)
                        # Use torch.empty_like + copy_ to create independent tensor
                        # This ensures no connection to unconditioned context computation graph
                        # Target shape is now [64, 4096] (truncated)
                        target_shape = (TARGET_ADAPTER_TOKENS, uncond_item.shape[1])  # [64, 4096]
                        target_dtype = uncond_item.dtype
                        target_device = uncond_item.device
                        
                        # Create new tensor with shape [64, 4096], dtype, and device
                        adapter_item = torch.empty(target_shape, dtype=target_dtype, device=target_device, requires_grad=False)
                        # Copy values from unconditioned context (detached) - truncated to 64 tokens
                        adapter_item.copy_(uncond_item.detach(), non_blocking=False)
                        
                        # FIX: Apply LayerNorm + learnable gate to unconditioned adapter output
                        # Note: This is detached, so no gradients, but still needs normalization/gating for consistency
                        adapter_item_normalized = self.adapter_output_norm(adapter_item)
                        adapter_item_gated = adapter_item_normalized * self.adapter_output_gate
                        adapter_item = adapter_item_gated
                    except Exception as e:
                        # Fallback to regular adapter output if unconditioned context extraction fails
                        logger.warning(f"Failed to extract unconditioned context: {e}, using regular adapter output")
                        if adapter_output is not None and isinstance(adapter_output, list) and adapter_output[idx] is not None:
                            if adapter_output[idx].dim() == 3:
                                # [1, K, 4096] -> [K, 4096] (truncate to 64)
                                adapter_item = adapter_output[idx][0]
                                if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                    adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                            else:
                                # Already [K, 4096] (truncate to 64)
                                adapter_item = adapter_output[idx]
                                if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                    adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                            
                            # FIX: Apply LayerNorm + learnable gate to fallback adapter output
                            if adapter_item is not None:
                                adapter_item_normalized = self.adapter_output_norm(adapter_item)
                                adapter_item_gated = adapter_item_normalized * self.adapter_output_gate
                                adapter_item = adapter_item_gated
                elif adapter_output is not None:
                    # FIX: Match OmniVideo processing - take [0] if dim==3 (same as OmniVideo line 312-313)
                    # CRITICAL: Truncate to 64 tokens to match OmniVideo's context size (~103 tokens total)
                    # This ensures context length is similar: 64 adapter + 77 T5 = 141 tokens (vs OmniVideo ~103)
                    # After retraining with adapter_query_length=64, this truncation won't be needed
                    TARGET_ADAPTER_TOKENS = 64  # Match OmniVideo's adapter output length
                    
                    if isinstance(adapter_output, list):
                        if adapter_output[idx] is not None:
                            if adapter_output[idx].dim() == 3:
                                # [B, K, 4096] -> [K, 4096] (K might be 256, truncate to 64)
                                adapter_item = adapter_output[idx][0]  # Match OmniVideo: take [0] if dim==3
                                # Truncate to 64 tokens to match OmniVideo
                                if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                    adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                                    if idx == 0:
                                        print(f"[DEBUG MobileOV] adapter_output[{idx}] shape: {adapter_output[idx].shape}, taking [0] -> truncated to {adapter_item.shape}", flush=True)
                                else:
                                    if idx == 0:
                                        print(f"[DEBUG MobileOV] adapter_output[{idx}] shape: {adapter_output[idx].shape}, taking [0] -> {adapter_item.shape}", flush=True)
                            else:
                                # Already [K, 4096] (2D)
                                adapter_item = adapter_output[idx]
                                # Truncate to 64 tokens if needed
                                if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                    adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                                    if idx == 0:
                                        print(f"[DEBUG MobileOV] adapter_output[{idx}] truncated: {adapter_output[idx].shape} -> {adapter_item.shape}", flush=True)
                                else:
                                    if idx == 0:
                                        print(f"[DEBUG MobileOV] adapter_output[{idx}] shape: {adapter_output[idx].shape} (2D)", flush=True)
                    elif adapter_output.dim() == 3:
                        # [B, K, 4096] -> [K, 4096]
                        adapter_item = adapter_output[idx][0]  # Match OmniVideo: take [0] if dim==3
                        # Truncate to 64 tokens if needed
                        if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                            adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                            if idx == 0:
                                print(f"[DEBUG MobileOV] adapter_output truncated: {adapter_output.shape} -> {adapter_item.shape}", flush=True)
                        else:
                            if idx == 0:
                                print(f"[DEBUG MobileOV] adapter_output shape: {adapter_output.shape}, taking [{idx}][0] -> {adapter_item.shape}", flush=True)
                    else:
                        # Already 2D [K, 4096]
                        adapter_item = adapter_output
                        # Truncate to 64 tokens if needed
                        if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                            adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                            if idx == 0:
                                print(f"[DEBUG MobileOV] adapter_output truncated: {adapter_output.shape} -> {adapter_item.shape}", flush=True)
                        else:
                            if idx == 0:
                                print(f"[DEBUG MobileOV] adapter_output shape: {adapter_output.shape} (2D)", flush=True)
                
                # Get context item (T5 embeddings)
                # Skip T5 context if disable_t5_context is True and we're using SmolVLM2
                # This avoids double-conditioning (T5 + SmolVLM2) which can confuse the model
                context_item = None
                if context is not None and not (self.disable_t5_context and not self.use_precomputed_features):
                    # Only use T5 context if:
                    # 1. disable_t5_context is False, OR
                    # 2. We're using precomputed features (not SmolVLM2)
                    if isinstance(context, list):
                        if context[idx].dim() == 3:
                            context_item = context[idx][0]
                        else:
                            context_item = context[idx]
                    elif context.dim() > 2:
                        context_item = context[idx]
                    else:
                        context_item = context
                    
                    # CRITICAL: T5 context is frozen (no gradients), so detach it to avoid gradient errors
                    # This prevents "element 0 of tensors does not require grad" error when backward
                    if context_item is not None and not context_item.requires_grad:
                        context_item = context_item.detach()
                
                # Get visual embeddings
                visual_item = None
                if processed_visual_emb is not None:
                    visual_item = processed_visual_emb[idx]
                
                # Get reference images
                ref_item = None
                if processed_ref_images is not None:
                    ref_item = processed_ref_images[idx]
                
                # Prepare context with special tokens
                if special_token_dict is not None:
                    if visual_item is not None and '<img_st>' in special_token_dict and '<img_ed>' in special_token_dict:
                        components.extend([
                            special_token_dict['<img_st>'],
                            visual_item,
                            special_token_dict['<img_ed>']
                        ])
                    
                    if ref_item is not None and '<img_st>' in special_token_dict and '<img_ed>' in special_token_dict:
                        components.extend([
                            special_token_dict['<img_st>'],
                            ref_item,
                            special_token_dict['<img_ed>']
                        ])
                    
                    if adapter_item is not None:
                        # CRITICAL FIX: Keep ALL 256 tokens from adapter output
                        # CRITICAL: NO .clone() - preserve gradients for training!
                        # adapter_item should be [256, 4096] (2D) - keep it as is!
                        # WanModel expects context to be List[Tensor] where each tensor is [L, C]
                        if adapter_item.dim() == 1:
                            # [4096] -> [1, 4096] (shouldn't happen, but handle it)
                            adapter_item = adapter_item.unsqueeze(0)  # No clone
                        elif adapter_item.dim() == 3:
                            # [1, 256, 4096] -> [256, 4096] (keep all 256 tokens)
                            adapter_item = adapter_item[0]  # Use indexing, not clone
                        elif adapter_item.dim() == 2:
                            # Already [256, 4096] - KEEP ALL 256 TOKENS! Don't take [0]!
                            # Use as is - no clone needed (preserve gradients)
                            pass
                        else:
                            # Fallback: flatten to 2D (preserve gradients)
                            adapter_item = adapter_item.view(-1, adapter_item.shape[-1])
                        if '<ipl_st>' in special_token_dict and '<ipl_ed>' in special_token_dict:
                            components.extend([
                                special_token_dict['<ipl_st>'],
                                adapter_item,
                                special_token_dict['<ipl_ed>']
                            ])
                        else:
                            components.append(adapter_item)
                    
                    if context_item is not None:
                        if '<prp_st>' in special_token_dict and '<prp_ed>' in special_token_dict:
                            components.extend([
                                special_token_dict['<prp_st>'],
                                context_item,
                                special_token_dict['<prp_ed>']
                            ])
                        else:
                            components.append(context_item)
                    
                    if components:
                        # CRITICAL: Ensure all components are detached if they don't require grad
                        # This prevents grad_fn errors when concatenating
                        normalized_components = []
                        for comp in components:
                            if isinstance(comp, torch.Tensor) and not comp.requires_grad:
                                # Create new tensor to break any potential graph connections
                                comp_new = torch.empty_like(comp)
                                comp_new.copy_(comp.detach(), non_blocking=False)
                                comp_new.requires_grad_(False)
                                normalized_components.append(comp_new)
                            else:
                                normalized_components.append(comp)
                        new_context = torch.cat(normalized_components, dim=0)
                        if self.max_context_len is not None and new_context.shape[0] > self.max_context_len:
                            new_context = new_context[0: self.max_context_len]
                    else:
                        raise ValueError("No components available to create context")
                else:
                    # Simple concatenation with gamma gating for clue/T5 mixing
                    items_to_concat = []
                    if visual_item is not None:
                        # Ensure visual_item is 2D [L, C]
                        if visual_item.dim() > 2:
                            visual_item = visual_item.view(-1, visual_item.shape[-1])  # Flatten to 2D
                        elif visual_item.dim() == 1:
                            visual_item = visual_item.unsqueeze(0)  # [C] -> [1, C]
                        items_to_concat.append(visual_item)
                    if ref_item is not None:
                        # Ensure ref_item is 2D [L, C]
                        if ref_item.dim() > 2:
                            ref_item = ref_item.view(-1, ref_item.shape[-1])  # Flatten to 2D
                        elif ref_item.dim() == 1:
                            ref_item = ref_item.unsqueeze(0)  # [C] -> [1, C]
                        items_to_concat.append(ref_item)
                    
                    # Full clue concatenation (no gamma gating)
                    # Concat: [full clue, T5] to match OmniVideo architecture
                    # Always use full adapter output for best results
                    if adapter_item is not None and context_item is not None:
                        # Both available: full concatenation (no gamma gating)
                        # FIX: Match OmniVideo processing - normalize to 2D [K, 4096] (K=64 for OmniVideo)
                        # Adapter item should already be 2D from previous processing, but ensure it's correct
                        if adapter_item.dim() == 1:
                            # [4096] -> [1, 4096] (shouldn't happen)
                            adapter_item = adapter_item.unsqueeze(0)
                        elif adapter_item.dim() == 3:
                            # [1, K, 4096] -> [K, 4096] (shouldn't happen if previous processing is correct)
                            adapter_item = adapter_item[0]
                        elif adapter_item.dim() == 2:
                            # Already [K, 4096] - use as is (K=64 for OmniVideo, 64 for MobileOV after fix)
                            pass  # Use as is
                        else:
                            # Fallback: flatten to 2D (preserve gradients)
                            adapter_item = adapter_item.view(-1, adapter_item.shape[-1])
                        
                        if context_item.dim() == 1:
                            context_item = context_item.unsqueeze(0)
                        elif context_item.dim() > 2:
                            if context_item.dim() == 3:
                                context_item = context_item[0]
                            else:
                                context_item = context_item.view(-1, context_item.shape[-1])
                        # else: already 2D, use as is
                        
                        # FIX: Apply LayerNorm + learnable gate to adapter output before concat
                        # This aligns adapter tokens to T5 distribution space and prevents scale mismatch
                        adapter_item_normalized = self.adapter_output_norm(adapter_item)
                        adapter_item_gated = adapter_item_normalized * self.adapter_output_gate
                        
                        # Full concatenation: [gated clue, T5] - with normalization and gating
                        # PROFILE: Log context sizes before concatenation
                        if idx == 0:
                            print(f"[PROFILE MobileOV] Before concat - adapter_item shape: {adapter_item.shape}, context_item shape: {context_item.shape}, gate={self.adapter_output_gate.item():.6f}", flush=True)
                        items_to_concat.append(adapter_item_gated)  # Gated and normalized clue
                        items_to_concat.append(context_item)  # Full T5
                        if idx == 0:
                            print(f"[DEBUG MobileOV] Items to concat: adapter={adapter_item.shape}, T5={context_item.shape}", flush=True)
                    elif adapter_item is not None:
                        # Only clue available (T5 disabled or not provided)
                        # FIX: Match OmniVideo processing - normalize to 2D [K, 4096]
                        # CRITICAL: Ensure adapter_item is truncated to 64 tokens (from earlier processing)
                        TARGET_ADAPTER_TOKENS = 64  # Match OmniVideo
                        
                        if adapter_item.dim() == 1:
                            # [4096] -> [1, 4096] (shouldn't happen)
                            adapter_item = adapter_item.unsqueeze(0)
                        elif adapter_item.dim() == 3:
                            # [1, K, 4096] -> [K, 4096] (truncate to 64)
                            adapter_item = adapter_item[0]
                            if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                        elif adapter_item.dim() == 2:
                            # Already [K, 4096] - truncate to 64 if needed
                            if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                        else:
                            # Fallback: flatten to 2D (preserve gradients)
                            adapter_item = adapter_item.view(-1, adapter_item.shape[-1])
                            if adapter_item.shape[0] > TARGET_ADAPTER_TOKENS:
                                adapter_item = adapter_item[:TARGET_ADAPTER_TOKENS, :]
                        
                        # FIX: Apply LayerNorm + learnable gate to adapter output
                        adapter_item_normalized = self.adapter_output_norm(adapter_item)
                        adapter_item_gated = adapter_item_normalized * self.adapter_output_gate
                        items_to_concat.append(adapter_item_gated)
                    elif context_item is not None:
                        # Only T5 available (clue not available) (preserve gradients)
                        if context_item.dim() == 1:
                            context_item = context_item.unsqueeze(0)
                        elif context_item.dim() > 2:
                            if context_item.dim() == 3:
                                context_item = context_item[0]
                            else:
                                context_item = context_item.view(-1, context_item.shape[-1])
                        items_to_concat.append(context_item)
                    
                    if items_to_concat:
                        # CRITICAL: Ensure all items are 2D [L, C] before concatenating
                        # WanModel expects context to be List[Tensor] where each tensor is [L, C]
                        # MATCH OmniVideo: Items are already normalized above, so we can concat directly
                        # However, we still need to handle gradients properly for training
                        # PROFILE: Context concatenation
                        import time as time_module
                        concat_start = time_module.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # MATCH OmniVideo: Direct concatenation (items already normalized)
                        # But preserve gradient handling for training compatibility
                        normalized_items = []
                        for i, item in enumerate(items_to_concat):
                            # Items should already be 2D from normalization above
                            # But double-check and handle gradients for training
                            if item.dim() == 1:
                                item = item.unsqueeze(0)
                            elif item.dim() > 2:
                                item = item.view(-1, item.shape[-1])
                            
                            # For training: Only detach items that truly don't have gradients
                            # This preserves gradient flow for trainable components (adapter_item)
                            if not item.requires_grad and item.grad_fn is None:
                                # Truly frozen tensor - safe to detach
                                item = item.detach().requires_grad_(False)
                            
                            normalized_items.append(item)
                        
                        # PROFILE: Log context sizes before concatenation (MATCH OmniVideo)
                        if idx == 0:
                            shapes_str = ", ".join([f"item{i}: {item.shape}" for i, item in enumerate(normalized_items)])
                            print(f"[PROFILE MobileOV] Before concat - {shapes_str}", flush=True)
                        
                        new_context = torch.cat(normalized_items, dim=0)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        concat_time = time_module.time() - concat_start
                        
                        # PROFILE: Log timing
                        if idx == 0:
                            print(f"[PROFILE MobileOV Step] Concatenation: {concat_time*1000:.2f}ms", flush=True)
                            print(f"[DEBUG MobileOV] Mixed context shape: {new_context.shape} (before truncate)", flush=True)
                        
                        # Truncate if needed (MATCH OmniVideo)
                        if self.max_context_len is not None and new_context.shape[0] > self.max_context_len:
                            old_len = new_context.shape[0]
                            new_context = new_context[0: self.max_context_len]
                            if idx == 0:
                                print(f"[DEBUG MobileOV] Truncated context: {old_len} -> {new_context.shape[0]} tokens", flush=True)
                        if idx == 0:
                            print(f"[DEBUG MobileOV] Final context shape: {new_context.shape}", flush=True)
                    else:
                        raise ValueError("No components available to create context")
                
                mixed_context.append(new_context)
                
                # FIX: Calculate context_lens for each sample in batch
                # This is critical for proper attention masking in WAN cross-attention
                # context_lens should be the actual length of context (after concat and truncate)
                if not hasattr(self, '_context_lens_list'):
                    self._context_lens_list = []
                context_len = new_context.shape[0]  # [L, C] -> L
                self._context_lens_list.append(context_len)
        else:
            raise ValueError(f"Condition mode {condition_mode} is not supported")
        
        # FIX: Create context_lens tensor for WAN model
        # WAN expects context_lens as tensor of shape [B] with actual context lengths
        if hasattr(self, '_context_lens_list') and len(self._context_lens_list) > 0:
            context_lens = torch.tensor(self._context_lens_list, dtype=torch.long, device=mixed_context[0].device)
            # Clear the list for next batch
            self._context_lens_list = []
        else:
            # Fallback: calculate from context shapes
            context_lens = torch.tensor([ctx.shape[0] for ctx in mixed_context], dtype=torch.long, device=mixed_context[0].device)
        
        # Forward through WanModel
        # PROFILE: WAN forward pass
        if len(mixed_context) > 0:
            print(f"[DEBUG MobileOV] WAN context input shape: {mixed_context[0].shape}, context_lens: {context_lens.tolist()}", flush=True)
        import time as time_module
        wan_start = time_module.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # FIX: Pass context_lens to WAN model for proper attention masking
        result = self.wan_model(x, t=t, context=mixed_context, seq_len=seq_len, context_lens=context_lens)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wan_time = time_module.time() - wan_start
        
        # PROFILE: Log timing and context info (log every time to see all steps)
        print(f"[PROFILE MobileOV] WAN forward pass: {wan_time*1000:.2f}ms", flush=True)
        if mixed_context and len(mixed_context) > 0:
            ctx_shape = mixed_context[0].shape if isinstance(mixed_context[0], torch.Tensor) else "N/A"
            print(f"[PROFILE MobileOV] Context shape: {ctx_shape}, Context length: {ctx_shape[0] if isinstance(ctx_shape, tuple) and len(ctx_shape) > 0 else 'N/A'}", flush=True)
        logger.info(f"[PROFILE] WAN forward pass: {wan_time*1000:.2f}ms")
        
        # Return adapter_output if requested (for distillation loss)
        if return_adapter_output:
            # Get adapter_output_batch from the forward pass
            # We need to store it during forward pass
            if hasattr(self, '_last_adapter_output_batch'):
                adapter_output_batch = self._last_adapter_output_batch
                # Truncate to 64 tokens if needed (match groundtruth)
                if adapter_output_batch.shape[1] > 64:
                    adapter_output_batch = adapter_output_batch[:, :64, :]  # [B, 64, 4096]
                return result, adapter_output_batch
            else:
                # Fallback: return None for adapter_output if not available
                return result, None
        
        return result
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for models"""
        self.wan_model.enable_gradient_checkpointing()
        if hasattr(self.adapter, 'enable_gradient_checkpointing'):
            self.adapter.enable_gradient_checkpointing()
        if self.smolvlm2_model is not None and hasattr(self.smolvlm2_model, 'enable_gradient_checkpointing'):
            self.smolvlm2_model.enable_gradient_checkpointing()
    
    def reset_wan_text_len(self, new_len: int):
        """Reset the text_len attribute of the WanModel"""
        self.wan_model.text_len = new_len
    
    def save_pretrained(self, save_dir: str):
        """Save the MobileOVModel to the specified directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save WanModel
        wan_save_dir = os.path.join(save_dir, "wan_model")
        os.makedirs(wan_save_dir, exist_ok=True)
        self.wan_model.save_pretrained(wan_save_dir)
        
        # Save adapter
        if self.adapter is not None:
            adapter_save_dir = os.path.join(save_dir, "adapter")
            os.makedirs(adapter_save_dir, exist_ok=True)
            adapter_state_dict = self.adapter.state_dict()
            torch.save(adapter_state_dict, os.path.join(adapter_save_dir, "adapter_pytorch_model.bin"))
        
        # Save visual context adapter
        if self.visual_context_adapter is not None:
            visual_adapter_save_dir = os.path.join(save_dir, "visual_context_adapter")
            os.makedirs(visual_adapter_save_dir, exist_ok=True)
            visual_adapter_state_dict = self.visual_context_adapter.state_dict()
            torch.save(visual_adapter_state_dict, os.path.join(visual_adapter_save_dir, "visual_context_adapter_pytorch_model.bin"))
        
        # Save SmolVLM2 projection
        if self.smolvlm2_projection is not None:
            projection_save_dir = os.path.join(save_dir, "smolvlm2_projection")
            os.makedirs(projection_save_dir, exist_ok=True)
            projection_state_dict = self.smolvlm2_projection.state_dict()
            torch.save(projection_state_dict, os.path.join(projection_save_dir, "smolvlm2_projection_pytorch_model.bin"))
        
        # Save SmolVLM2VisionHead
        if self.smolvlm2_vision_head is not None:
            vh_save_dir = os.path.join(save_dir, "smolvlm2_vision_head")
            os.makedirs(vh_save_dir, exist_ok=True)
            self.smolvlm2_vision_head.save_pretrained(vh_save_dir)
        
        logger.info(f"Model saved to {save_dir}")
    
    def to(self, *args, **kwargs):
        """Move the model to the specified device"""
        self.wan_model = self.wan_model.to(*args, **kwargs)
        if self.adapter is not None:
            self.adapter = self.adapter.to(*args, **kwargs)
        if self.smolvlm2_model is not None:
            self.smolvlm2_model = self.smolvlm2_model.to(*args, **kwargs)
        if self.smolvlm2_projection is not None:
            self.smolvlm2_projection = self.smolvlm2_projection.to(*args, **kwargs)
        if self.smolvlm2_vision_head is not None:
            self.smolvlm2_vision_head = self.smolvlm2_vision_head.to(*args, **kwargs)
        if self.visual_context_adapter is not None:
            self.visual_context_adapter = self.visual_context_adapter.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def enable_train(self):
        """Enable training mode for all models"""
        self.wan_model.train()
        if self.adapter is not None:
            self.adapter.train()
        if self.smolvlm2_model is not None:
            self.smolvlm2_model.train()
        if self.smolvlm2_projection is not None:
            self.smolvlm2_projection.train()
        if self.smolvlm2_vision_head is not None:
            self.smolvlm2_vision_head.train()
        if self.visual_context_adapter is not None:
            self.visual_context_adapter.train()
    
    def enable_eval(self):
        """Enable evaluation mode for all models"""
        self.wan_model.eval()
        if self.adapter is not None:
            self.adapter.eval()
        if self.smolvlm2_model is not None:
            self.smolvlm2_model.eval()
        if self.smolvlm2_projection is not None:
            self.smolvlm2_projection.eval()
        if self.smolvlm2_vision_head is not None:
            self.smolvlm2_vision_head.eval()
        if self.visual_context_adapter is not None:
            self.visual_context_adapter.eval()
    
    def is_train(self):
        """Check if the model is in training mode"""
        is_training = self.wan_model.training
        if self.adapter is not None:
            is_training = is_training and self.adapter.training
        if self.visual_context_adapter is not None:
            is_training = is_training and self.visual_context_adapter.training
        return is_training
    
    def understand_image_or_video(
        self,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        query: str = "Describe this image/video in detail.",
        max_new_tokens: Optional[int] = None,  # None = no limit (only stop at EOS), but will use hard limit
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        device: Optional[torch.device] = None,
        max_length_hard_limit: int = 2048,  # Hard limit to prevent infinite generation
    ) -> str:
        """
        Perform image/video understanding using SmolVLM2.
        
        Args:
            image_path: Path to image file (jpg, png)
            video_path: Path to video file (mp4)
            query: Question or prompt for understanding
            max_new_tokens: Maximum tokens to generate. If None, will generate until EOS token 
                           (with hard limit of max_length_hard_limit to prevent infinite generation)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to run on (uses model device if None)
            max_length_hard_limit: Hard limit for total sequence length (input + generated) 
                                  when max_new_tokens is None
            
        Returns:
            Generated text description/answer
            
        Raises:
            RuntimeError: If model doesn't support text generation
        """
        if self.smolvlm2_model is None:
            raise RuntimeError("SmolVLM2 model not initialized")
        
        # Check if model has generate capability
        if not hasattr(self.smolvlm2_model, '_model'):
            raise RuntimeError("SmolVLM2 model not loaded properly")
        
        # Check if underlying model has generate method
        has_generate = False
        if hasattr(self.smolvlm2_model, 'generate'):
            has_generate = True
        elif hasattr(self.smolvlm2_model._model, 'generate'):
            has_generate = True
        
        if not has_generate:
            raise RuntimeError(
                "SmolVLM2 model does not have generate() method. "
                "Please load SmolVLMForConditionalGeneration checkpoint with lm_head."
            )
        
        # Get device
        if device is None:
            device = next(self.smolvlm2_model.parameters()).device
        
        # Get tokenizer (with caching, same as encode_prompts_with_smolvlm2)
        tokenizer = self.smolvlm2_model.get_tokenizer()
        if tokenizer is None:
            # Fallback: Try to load tokenizer from HuggingFace (cache it)
            if not hasattr(self, '_cached_tokenizer'):
                logger.warning("Tokenizer not found in checkpoint, trying to load from HuggingFace...")
                try:
                    from transformers import AutoTokenizer
                    # SmolVLM2 model name on HuggingFace
                    self._cached_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
                    logger.info("✓ Loaded tokenizer from HuggingFace (cached)")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer from HuggingFace: {e}")
                    raise RuntimeError("Tokenizer not available in SmolVLM2 model and HuggingFace fallback failed")
            tokenizer = self._cached_tokenizer
        
        # Load and process image/video if provided
        pixel_values = None
        if image_path is not None and os.path.exists(image_path):
            try:
                import PIL.Image
                from nets.third_party.llava.mm_utils import process_images
                
                # Load image
                image = PIL.Image.open(image_path).convert("RGB")
                images = [image]
                
                # Get image processor from model config if available
                # For now, we'll use a simple approach: just tokenize text
                # TODO: Add proper image processing when processor is available
                logger.warning("Image processing not fully implemented yet. Using text-only mode.")
                pixel_values = None
            except Exception as e:
                logger.warning(f"Failed to load image: {e}. Using text-only mode.")
                pixel_values = None
        elif video_path is not None and os.path.exists(video_path):
            try:
                from nets.third_party.llava.mm_utils import opencv_extract_frames, process_images
                
                # Extract frames
                images, _ = opencv_extract_frames(video_path, num_frames=8)
                
                # TODO: Process frames properly when processor is available
                logger.warning("Video processing not fully implemented yet. Using text-only mode.")
                pixel_values = None
            except Exception as e:
                logger.warning(f"Failed to load video: {e}. Using text-only mode.")
                pixel_values = None
        
        # For now, use text-only generation (simpler and works without processor)
        # Tokenize query
        inputs = tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Determine max_new_tokens: if None, calculate from hard limit
        input_length = input_ids.shape[1]
        if max_new_tokens is None:
            # Generate until EOS, but respect hard limit
            max_new_tokens = max_length_hard_limit - input_length
            if max_new_tokens <= 0:
                max_new_tokens = 512  # Fallback
            logger.info(f"max_new_tokens=None: will generate up to {max_new_tokens} tokens (hard limit: {max_length_hard_limit}, input length: {input_length})")
        else:
            logger.info(f"Using max_new_tokens={max_new_tokens}")
        
        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else (tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None),
        }
        
        # Always add EOS token to allow natural stopping (model will stop when it finishes)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
            logger.info(f"EOS token enabled: {tokenizer.eos_token_id} (model will stop naturally when finished)")
        else:
            logger.warning("No EOS token found - generation will continue until max_new_tokens")
        
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        
        # Add repetition penalty to avoid repetition
        generation_kwargs["repetition_penalty"] = 1.1
        
        # Generate text
        try:
            # Ensure model is in eval mode and on correct device
            if hasattr(self.smolvlm2_model, '_model'):
                self.smolvlm2_model._model.eval()
                # Ensure model is on correct device and dtype
                self.smolvlm2_model._model = self.smolvlm2_model._model.to(device)
                # Ensure lm_head has same dtype as model
                if hasattr(self.smolvlm2_model._model, 'lm_head'):
                    model_dtype = next(self.smolvlm2_model._model.parameters()).dtype
                    if self.smolvlm2_model._model.lm_head.weight.dtype != model_dtype:
                        self.smolvlm2_model._model.lm_head = self.smolvlm2_model._model.lm_head.to(dtype=model_dtype)
            
            with torch.no_grad():
                if hasattr(self.smolvlm2_model, 'generate'):
                    # Use wrapper's generate method
                    generated_ids = self.smolvlm2_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        **generation_kwargs
                    )
                elif hasattr(self.smolvlm2_model, '_model') and hasattr(self.smolvlm2_model._model, 'generate'):
                    # Use underlying model's generate method
                    generated_ids = self.smolvlm2_model._model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        **generation_kwargs
                    )
                else:
                    raise RuntimeError("Model does not have generate() method")
            
            # Decode generated text
            # Remove input tokens from output
            generated_ids_trimmed = generated_ids[0][input_ids.shape[1]:]
            
            # Check if generation stopped early (before max_new_tokens)
            num_generated = generated_ids_trimmed.shape[0]
            logger.info(f"Generated {num_generated} tokens (requested max_new_tokens={max_new_tokens})")
            
            # Check for EOS token in generated sequence
            eos_found = False
            eos_position = None
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                eos_positions = (generated_ids_trimmed == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_found = True
                    eos_position = eos_positions[0].item()
                    # Truncate at EOS token
                    generated_ids_trimmed = generated_ids_trimmed[:eos_position]
                    logger.info(f"✅ EOS token found at position {eos_position}, truncating output")
                else:
                    logger.info("⚠️  No EOS token found in generated sequence (model may not generate EOS naturally)")
            
            if num_generated < max_new_tokens:
                if eos_found:
                    logger.info(f"Generation stopped early at token {eos_position} due to EOS token")
                else:
                    logger.info("Generation stopped early but no EOS token found (may be model limit or other stopping criteria)")
            else:
                if eos_found:
                    logger.info(f"EOS token found but generation also reached max_new_tokens limit ({max_new_tokens})")
                else:
                    logger.info(f"Generation reached max_new_tokens limit ({max_new_tokens}) without EOS token")
            
            generated_text = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
            generated_text = generated_text.strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError(f"Failed to generate text: {e}")

