# Mobile-OV Model Architecture

## Tổng quan

`MobileOVModel` là một architecture thống nhất tích hợp **SmolVLM2-500M** làm understanding module và **WAN** làm generation module.

### So sánh với OmniVideoMixedConditionModel

| Component | OmniVideoMixedConditionModel | MobileOVModel |
|-----------|------------------------------|---------------|
| Understanding | VisionHead (pre-computed features) | SmolVLM2-500M (on-the-fly encoding) |
| Generation | WAN | WAN (giữ nguyên) |
| Features | Cần pre-compute `vlm_last_hidden_states` | Có thể encode prompts trực tiếp |

## Cách sử dụng

### 1. Config YAML

Thêm vào config YAML của bạn:

```yaml
training:
  model_settings:
    model_type: "mobile_ov"  # hoặc "omnivideo" để dùng model gốc
    smolvlm2_ckpt_path: "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"
    use_precomputed_features: false  # true = dùng pre-computed, false = encode on-the-fly
    train_smolvlm2: false  # có train SmolVLM2 không
    train_smolvlm2_projection: true  # có train projection layer không
```

### 2. Chế độ hoạt động

#### Mode 1: Pre-computed features (backward compatible)
```yaml
use_precomputed_features: true
```
- Giống như OmniVideoMixedConditionModel
- Đọc `vlm_last_hidden_states` từ dataset
- Không cần prompts trong forward pass

#### Mode 2: On-the-fly encoding (unified architecture)
```yaml
use_precomputed_features: false
```
- Encode prompts trực tiếp bằng SmolVLM2 trong forward pass
- Không cần pre-compute features
- End-to-end training có thể

### 3. Training

```bash
python finetune_model.py --config configs/mobile_ov_config.yaml
```

## Architecture Details

### Components

1. **SmolVLM2-500M**: Understanding module
   - Encode text prompts → hidden states
   - Output: `[B, L, 1024]` (SmolVLM2 hidden size)

2. **Projection Layer**: Map SmolVLM2 output to adapter input
   - `Linear(1024, 1152)` để map từ SmolVLM2 → adapter input size

3. **DM_Adapter**: Giữ nguyên từ Omni-Video
   - Input: `[B, L, 1152]`
   - Output: `[B, L, 4096]`

4. **WAN Model**: Generation module (giữ nguyên)

### Forward Pass Flow

```
Input: prompts (text) + videos
  ↓
SmolVLM2.encode_prompts_with_smolvlm2()
  → [B, L, 1024]
  ↓
smolvlm2_projection
  → [B, L, 1152]
  ↓
DM_Adapter
  → [B, L, 4096]
  ↓
WAN Model (với mixed context)
  → Generated video
```

## Lưu ý

1. **Hidden size**: SmolVLM2-500M có hidden_size=1024, cần projection để map lên 1152 (adapter input)
2. **Training**: Mặc định SmolVLM2 được freeze, chỉ train projection layer. Set `train_smolvlm2: true` để train cả SmolVLM2
3. **Memory**: On-the-fly encoding tốn memory hơn pre-computed, nhưng cho phép end-to-end training

## Roadmap

- [x] Experiment 1: SmolVLM2 thay understanding module ✅
- [ ] Experiment 2: SANA-video thay generation module
- [ ] Experiment 3: SmolVLM2 + SANA unified model

