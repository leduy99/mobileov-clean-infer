# Mobile-OV Modules

This directory contains the active Mobile-OV generation modules used by the
clean inference repo.

Files:

- `mobile_ov_model.py`
  The top-level full Mobile-OV inference model with generation and
  understanding methods.
- `mobile_ov_bridge.py`
  The main bridge from SmolVLM2 hidden states into the SANA conditioning space.
- `adapter.py`
  Helper adapter block kept because the bridge constructor still references it.
- `smolvlm2_vision_head.py`
  Helper resampling block kept for completeness, even though the current
  generation path does not use the vision-head branch.

If you are reading the code in order, start here:

1. `generate.py`
2. `nets/mobile_ov/mobile_ov_model.py`
3. `nets/mobile_ov/mobile_ov_bridge.py`
