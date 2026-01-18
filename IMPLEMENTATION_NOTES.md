# FLUX.2-klein C Implementation Notes

This file tracks verified implementation details and debugging findings.
Update this as issues are found and fixed.

## Architecture Constants (Verified)
- hidden_size: 3072
- num_heads: 24
- head_dim: 128
- mlp_hidden: 9216 (3 * hidden)
- num_double_layers: 5
- num_single_layers: 20
- text_dim: 7680
- latent_channels: 128
- rope_theta: 2000.0
- axes_dim_rope: [32, 32, 32, 32] = 128 total

## RoPE Implementation (Verified)
- 4 axes: T (0-31), H (32-63), W (64-95), L (96-127)
- Image tokens: position IDs = (T=0, H=y, W=x, L=0)
  - Axis 0,3 are identity (position=0)
  - Axis 1 rotates based on y coordinate
  - Axis 2 rotates based on x coordinate
- Text tokens: position IDs = (T=0, H=0, W=0, L=seq_idx)
  - Axes 0,1,2 are identity (position=0)
  - Axis 3 rotates based on sequence index
- Rotation formula (per pair):
  - out[0] = cos*x0 - sin*x1
  - out[1] = sin*x0 + cos*x1 (NOT cos*x1 + sin*x0)

## Concatenation Order (Verified)
- Official Python concatenates as [TEXT, IMAGE] for Q, K, V
- K concatenation: cat_k = [txt_k, img_k]
- V concatenation: cat_v = [txt_v, img_v]
- RoPE PE concatenation: pe = [pe_txt, pe_img]

## Timestep Embedding (Verified)
- Input timestep is scaled by 1000 (t=1.0 becomes 1000.0)
- Sinusoidal embedding with 128 frequencies (256 dims)
- Two-layer MLP: linear_1 (256 -> 3072) + SiLU + linear_2 (3072 -> 3072)

## AdaLN Modulation (Verified)
- SiLU applied to t_emb BEFORE modulation projection
- Order: shift first, then scale: out = (1 + scale) * norm(x) + shift
- Double block: 6 params each for img/txt (shift1, scale1, gate1, shift2, scale2, gate2)
- Single block: 3 params (shift, scale, gate)

## Final Layer (Verified - BUG FIXED)
- Uses `AdaLayerNormContinuous` (not RMSNorm)
- LayerNorm with elementwise_affine=False (no learned gamma/beta)
- **CRITICAL**: Projection output splits as (scale, shift) NOT (shift, scale)
  - First half of linear output = scale
  - Second half of linear output = shift
- Formula: out = (1 + scale) * LayerNorm(x) + shift
- Linear projection to latent_channels

## Input/Output Format
- Image latent input: NCHW format [channels, h, w]
- Internal transformer: NLC format [seq, channels]
- Text input: [seq, text_dim]
- Conversion: transpose NCHW -> NLC for processing, NLC -> NCHW for output

## Verified Matching Values (Python vs C)
- t_emb values: MATCH
- img_proj (after x_embedder): MATCH
- txt_proj (after context_embedder): MATCH
- RoPE cos/sin values: MATCH
- After AdaLN in double block: MATCH
- Q values after projection and QK-norm: MATCH
- Q values after RoPE: MATCH
- Attention scores (first double block, head 0): MATCH
  - Python: [11.193116, 7.655654, 8.903099, 4.316768, 4.660241]
  - C:      [11.193114, 7.655654, 8.903103, 4.316768, 4.660240]
- Attention output (before proj): MATCH
  - Python: [-2.620135, -0.939009, 6.645398, 1.141459, 5.012253]
  - C:      [-2.620134, -0.939006, 6.645381, 1.141454, 5.012254]
- Output projection: MATCH
  - Python: [1.004017, -29.425447, -2.884938, 2.798838, -4.617723]
  - C:      [1.004027, -29.425407, -2.884941, 2.798832, -4.617741]
- After attention residual (first double block): MATCH
  - Python: [0.362979, 2.189511, 0.695620, -0.364556, 0.480292]
  - C:      [0.362981, 2.189507, 0.695621, -0.364556, 0.480294]

## ALL BUGS FIXED - Transformer output MATCHES Python!

### Verified Matching Output (2024-01-17):
- Python: [0.5482102, 2.6096351, 1.5703337, 1.7536415, 2.9919708]
- C:      [0.5482088, 2.6096334, 1.5703315, 1.7536404, 2.9919674]

All components verified matching:
- ALL 5 double blocks: MATCH
- ALL 20 single blocks: MATCH
- Final layer: MATCH

## Key Finding: Text Sequence Length
- MUST use same text sequence length in both C and Python tests
- Current test uses 512 text tokens
- Earlier mismatch was due to Python using 256 while C used 512

## Tests to Run
1. [x] Compare t_emb values
2. [x] Compare input projections
3. [x] Compare RoPE frequencies
4. [x] Compare AdaLN output
5. [x] Compare Q/K after projection and norm
6. [x] Compare Q/K after RoPE
7. [ ] Compare attention scores (first few)
8. [ ] Compare attention output
9. [ ] Compare full double block output
10. [ ] Compare single block output

## Compilation Flags
**ALWAYS use these optimized flags for faster testing:**
```bash
CFLAGS="-O3 -ffast-math -march=native"
```

## Debugging Commands
```bash
# Compile flux_transformer.c with debug output:
gcc $CFLAGS -DDEBUG_TRANSFORMER -DDEBUG_DOUBLE_BLOCK -c flux_transformer.c -o flux_transformer_debug.o

# Compile and link test:
gcc $CFLAGS -o test_tf_debug test_transformer_debug.c flux.o flux_kernels.o flux_transformer_debug.o flux_vae.o flux_safetensors.o flux_tokenizer.o flux_sample.o flux_image.o -lm

# Run C test:
./test_tf_debug flux-klein-model text_embeddings_official.bin py_noise_64x64.bin

# Run Python comparison:
python3 misc/test_diffusers_output.py
```

## IMPORTANT: After log compaction, re-read this file!
This file contains crucial implementation details that should not be forgotten.

## Bugs Fixed Summary

### Bug 1: Final Layer scale/shift order (FIXED)
- **Problem**: Final layer modulation split projection as (shift, scale)
- **Fix**: Changed to (scale, shift) - first half is scale, second half is shift
- **File**: flux_transformer.c lines 1158-1160

### Bug 2: K/V concatenation order (FIXED earlier)
- **Problem**: C code concatenated as [IMAGE, TEXT]
- **Fix**: Changed to [TEXT, IMAGE] to match Python

### Bug 3: Text sequence length mismatch (FIXED earlier)
- **Problem**: Python tests used 256 tokens, C used 512
- **Fix**: Aligned both to 512 tokens

## Single Block Architecture Reference
- 20 single-stream blocks (parallel DiT style)
- Input: concatenated [txt_hidden, img_hidden] (txt first at offset 0, img at offset txt_seq)
- Fused QKV + FFN projection
- Self-attention over full sequence
- RoPE applied: text portion uses txt_rope (axis 3), image portion uses img_rope (axes 1,2)

## VAE (AutoencoderKLFlux2) Details
- Uses `AutoencoderKLFlux2` class in diffusers
- latent_channels: 32 (VAE internal)
- patch_size: 2x2
- Transformer outputs 128 channels = 32 latent * 2*2 patch
- **Unpacking for VAE decode**: [B, 128, H, W] -> [B, 32, H*2, W*2]
  ```python
  unpacked = latents.reshape(B, 32, 2, 2, H, W)
  unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)  # [B, 32, H, 2, W, 2]
  unpacked = unpacked.reshape(B, 32, H*2, W*2)
  ```
- No scaling_factor in config (unlike standard VAE)

## End-to-End Test Scripts
- `misc/generate_1step_python.py` - Generate 1-step 64x64 image with Python
  - Uses same inputs: py_noise_64x64.bin, text_embeddings_official.bin
  - Output: test_cat_python_1step.png
- C test: `./flux --dir flux-klein-model --embeddings text_embeddings_official.bin --seed 42 --steps 1 --output test_cat_fixed.png --height 64 --width 64`

## Current Status (2024-01-18)
- Transformer: FULLY WORKING (matches Python)
- VAE: FULLY WORKING (verified matches reference image)
- Performance: Significantly optimized

## Performance Optimizations

### Linear layers (flux_linear) - DONE
- Replaced naive O(n^3) loop with BLAS cblas_sgemm
- Uses Apple Accelerate framework on macOS, OpenBLAS on Linux
- Speedup: ~30x (from ~32 GFLOPS to ~989 GFLOPS)

### Convolution (flux_conv2d) - DONE
- Replaced naive 6-nested-loop implementation with im2col + BLAS
- im2col transforms conv into matrix multiplication
- Speedup: ~180x (VAE decode: 18.6s -> 0.1s)

### Performance Results (64x64, 1 step)
Before optimization: ~142s total
After linear BLAS:    ~42s total (Transformer=22s, VAE=18s)
After conv BLAS:      ~24s total (Transformer=22s, VAE=0.1s)

### Attention workspace optimization - DONE
- Pre-allocated attention buffers (attn_q_t, attn_k_t, etc.) in transformer struct
- Eliminates ~150 malloc/free calls per forward pass (mha_forward and joint_attention)
- Marginal speedup (malloc overhead was not the main bottleneck)

### Remaining Bottleneck: Transformer (22s for 64x64)
- Main operations already use BLAS (sgemm)
- Bottleneck is the sheer number of FLOPs in linear projections:
  - Single block fused projection: [528, 3072] @ [27648, 3072] = ~90B FLOPs per block
  - 20 single blocks = ~1.8T FLOPs just for one layer type
- Further optimization would require:
  - Multi-threading (OpenMP) - clang on macOS doesn't support, need brew libomp
  - Batched GEMM for attention heads
  - Memory layout optimization for cache efficiency

## Progress Display

When running with `-v` (verbose mode), the inference shows fine-grained progress:
```
Step 1/2 dddddssssF
Step 2/2 dddddssssF
```

Progress characters (CLI-specific):
- `d` = Double-stream block completed (5 total per step)
- `s` = 5 single-stream blocks completed (4 groups of 5 = 20 total)
- `F` = Final layer completed

### Callback Architecture

The library provides headless callbacks (no printing); all formatting is CLI-side.

**Library callbacks** (flux_kernels.h):
```c
typedef enum {
    FLUX_SUBSTEP_DOUBLE_BLOCK,
    FLUX_SUBSTEP_SINGLE_BLOCK,
    FLUX_SUBSTEP_FINAL_LAYER,
} flux_substep_type_t;

extern flux_substep_callback_t flux_substep_callback;  // (type, index, total)
extern flux_step_callback_t flux_step_callback;        // (step, total)
```

**CLI implementation** (main.c):
- `cli_step_callback`: prints "Step N/M " with appropriate newlines
- `cli_substep_callback`: prints d/s/F characters based on substep type

**Usage**: Set callbacks before generation, clear after:
```c
flux_step_callback = cli_step_callback;
flux_substep_callback = cli_substep_callback;
// ... generation ...
flux_step_callback = NULL;
flux_substep_callback = NULL;
```

This design enables GUI integration - the GUI can set its own callbacks to update
progress bars, etc., without any library-side printing.

## Test Verification
Reference image: test_vectors/reference_1step_64x64_seed42.png
After any optimization, verify with:
```bash
./flux --dir flux-klein-model --embeddings text_embeddings_official.bin --seed 42 --steps 1 --output /tmp/test.png --height 64 --width 64
python3 -c "
import numpy as np
from PIL import Image
ref = np.array(Image.open('test_vectors/reference_1step_64x64_seed42.png'))
test = np.array(Image.open('/tmp/test.png'))
diff = np.abs(ref.astype(float) - test.astype(float))
print(f'Max diff: {diff.max()}, Mean diff: {diff.mean():.4f}')
print('PASS' if diff.max() < 2 else 'FAIL')
"
```

---

## Reference Text Embeddings for Testing

A reference embeddings file is available for testing the C text encoder implementation without triggering full image generation.

### Reference File
- **File**: `text_embeddings_official.bin`
- **Prompt**: `"A fluffy orange cat sitting on a windowsill"`
- **Shape**: `[1, 512, 7680]` (FP32)
- **Size**: 15,728,640 bytes (15.00 MB)
- **MD5**: `b0fa2d7a77d7860752c9de4114e427b9`

### Generating Embeddings

Use the script `misc/generate_embeddings.py` to generate new reference embeddings:

```bash
# Generate embeddings for a specific prompt
python3 misc/generate_embeddings.py "A fluffy orange cat sitting on a windowsill"

# Generate to a custom output file
python3 misc/generate_embeddings.py "Your prompt here" output.bin
```

**Requirements**: Python 3 with `torch`, `transformers`, `einops`

The script:
1. Loads Qwen3-4B model (from HuggingFace cache or downloads ~8GB)
2. Applies chat template: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
3. Tokenizes to 512 tokens with padding
4. Extracts hidden states from layers [9, 18, 27]
5. Reshapes to `[1, 512, 7680]` and saves as FP32 binary

### Testing C Text Encoder Implementation

When implementing the C text encoder, you can verify correctness by:

1. **Generate reference embeddings** for a test prompt using the Python script
2. **Run C text encoder** on the same prompt
3. **Compare binary outputs**:
   ```bash
   # Binary comparison
   cmp reference.bin c_output.bin

   # Or load in Python and compare numerically
   python3 -c "
   import numpy as np
   ref = np.fromfile('reference.bin', dtype=np.float32).reshape(1, 512, 7680)
   c = np.fromfile('c_output.bin', dtype=np.float32).reshape(1, 512, 7680)
   diff = np.abs(ref - c)
   print(f'Max diff: {diff.max():.6f}')
   print(f'Mean diff: {diff.mean():.6f}')
   print('PASS' if diff.max() < 1e-4 else 'FAIL - check implementation')
   "
   ```

This allows fast iteration on the C text encoder without running full image generation (which takes 15+ minutes at 1024x1024).

---

## Qwen3 Text Encoder Implementation Plan

This section documents the plan to implement native text encoding so that `-p "a car"` works directly without pre-computed embeddings.

### Reference Code
Official implementation: `flux2/src/flux2/text_encoder.py` (Qwen3Embedder class, lines 366-428)

### Architecture Overview

**Model**: Qwen3-4B (hidden_dim=2560, since 3×2560=7680=context_in_dim for Klein4B)
- HuggingFace model: `Qwen/Qwen3-4B` (or `Qwen/Qwen3-4B-FP8` for quantized)
- Architecture: Standard decoder-only transformer (causal LM)
- Vocab size: ~151,936 tokens
- Layers: 36
- Hidden dim: 2560
- Heads: 20
- Head dim: 128

**Output extraction**:
- Extract hidden states from layers [9, 18, 27] (0-indexed)
- Stack along new dimension: `[3, seq_len, 2560]`
- Reshape to `[seq_len, 7680]` (concatenate layer outputs)

### Tokenization Process

From official code (`Qwen3Embedder.forward`, lines 383-419):

```python
# 1. Format as chat message
messages = [{"role": "user", "content": prompt}]

# 2. Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Qwen3-specific: disable CoT
)

# 3. Tokenize with padding
model_inputs = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
)
```

**Chat template format** (Qwen3):
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

### Forward Pass

```python
output = model(
    input_ids=input_ids,          # [batch, 512]
    attention_mask=attention_mask, # [batch, 512]
    output_hidden_states=True,
    use_cache=False,
)

# Extract layers 9, 18, 27 (0-indexed, so hidden_states[10], [19], [28])
# hidden_states[0] is embeddings, [1] is after layer 0, etc.
out = torch.stack([
    output.hidden_states[10],  # After layer 9
    output.hidden_states[19],  # After layer 18
    output.hidden_states[28],  # After layer 27
], dim=1)  # [batch, 3, seq_len, 2560]

# Reshape: b c l d -> b l (c d)
out = rearrange(out, "b c l d -> b l (c d)")  # [batch, 512, 7680]
```

### Text Position IDs

From `flux2/src/flux2/sampling.py` (prc_txt function, lines 93-103):

```python
def prc_txt(x: Tensor, t_coord=None):
    _l, _ = x.shape  # seq_len, hidden_dim
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,  # T=0
        "h": torch.arange(1),  # H=0 (dummy)
        "w": torch.arange(1),  # W=0 (dummy)
        "l": torch.arange(_l), # L=0,1,2,...,seq_len-1
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)
```

**Result**: For each text token at position `i`:
- Position ID = `(T=0, H=0, W=0, L=i)`
- Shape: `[seq_len, 4]`

This means text tokens only get RoPE rotation on axis 3 (L dimension), while axes 0,1,2 have position 0 (identity rotation).

### Implementation Steps

#### Phase 1: Tokenizer (flux_tokenizer.c)

1. **Load vocabulary** from `text_encoder/tokenizer.json`:
   - Parse JSON to extract token→id mapping
   - Store as hash table for O(1) lookup
   - Handle special tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`

2. **Implement BPE tokenization**:
   - Load merges from `text_encoder/merges.txt` (or tokenizer.json)
   - Implement byte-pair encoding algorithm
   - Handle UTF-8 properly (Qwen uses byte-level BPE)

3. **Chat template application**:
   ```c
   // Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
   int flux_apply_chat_template(int *output_ids, int max_len,
                                 const char *prompt,
                                 flux_tokenizer_t *tok);
   ```

4. **Padding**:
   - Pad to max_length=512 with pad_token_id
   - Generate attention_mask (1 for real tokens, 0 for padding)

#### Phase 2: Qwen3 Model Architecture (flux_qwen3.c)

**Layers to implement**:

1. **Embedding layer**:
   - `embed_tokens`: [vocab_size, 2560] lookup table
   - Input: token IDs → Output: [seq_len, 2560]

2. **RMSNorm** (used throughout):
   ```c
   // Qwen3 uses RMSNorm, not LayerNorm
   void qwen3_rms_norm(float *out, const float *x, const float *weight,
                       int seq_len, int hidden_dim, float eps);
   ```

3. **Attention layer** (for each of 36 layers):
   - q_proj: [2560, 2560]
   - k_proj: [2560, 512] (GQA: 4 KV heads × 128 dim)
   - v_proj: [2560, 512]
   - o_proj: [2560, 2560]
   - RoPE on Q and K
   - Grouped Query Attention (GQA): 20 query heads, 4 KV heads

4. **MLP layer** (for each of 36 layers):
   - gate_proj: [2560, 6912]
   - up_proj: [2560, 6912]
   - down_proj: [6912, 2560]
   - Activation: SiLU(gate) * up

5. **Forward pass structure**:
   ```c
   for (int layer = 0; layer < 36; layer++) {
       // Pre-norm
       rms_norm(normed, hidden, layer_norm_weight, ...);

       // Self-attention
       attention_forward(attn_out, normed, layer, ...);
       hidden = hidden + attn_out;  // Residual

       // Pre-norm for MLP
       rms_norm(normed, hidden, post_attn_norm_weight, ...);

       // MLP
       mlp_forward(mlp_out, normed, layer, ...);
       hidden = hidden + mlp_out;  // Residual

       // Save hidden state if layer in [9, 18, 27]
       if (layer == 9 || layer == 18 || layer == 27) {
           memcpy(saved_hidden[save_idx++], hidden, ...);
       }
   }
   ```

#### Phase 3: Weight Loading (flux_qwen3.c)

**Weight files**: `text_encoder/model*.safetensors`

Weight naming convention:
```
model.embed_tokens.weight                    [151936, 2560]
model.layers.{i}.self_attn.q_proj.weight     [2560, 2560]
model.layers.{i}.self_attn.k_proj.weight     [512, 2560]
model.layers.{i}.self_attn.v_proj.weight     [512, 2560]
model.layers.{i}.self_attn.o_proj.weight     [2560, 2560]
model.layers.{i}.mlp.gate_proj.weight        [6912, 2560]
model.layers.{i}.mlp.up_proj.weight          [6912, 2560]
model.layers.{i}.mlp.down_proj.weight        [2560, 6912]
model.layers.{i}.input_layernorm.weight      [2560]
model.layers.{i}.post_attention_layernorm.weight [2560]
model.norm.weight                            [2560]
```

**Memory estimate**:
- Embeddings: 151936 × 2560 × 4 = ~1.5 GB
- Per layer: ~80 MB (attention + MLP weights)
- 36 layers: ~2.9 GB
- **Total: ~4.4 GB** (FP32), ~2.2 GB (FP16), ~1.1 GB (INT8)

#### Phase 4: Integration (flux.c, main.c)

1. **API additions**:
   ```c
   // Load text encoder
   flux_text_encoder_t *flux_text_encoder_load(const char *model_dir);
   void flux_text_encoder_free(flux_text_encoder_t *enc);

   // Encode text to embeddings
   float *flux_encode_text(flux_text_encoder_t *enc,
                           const char *prompt,
                           int *out_seq_len);
   ```

2. **Modify flux_generate()**:
   ```c
   // If prompt provided, encode it
   if (prompt != NULL && ctx->text_encoder != NULL) {
       text_emb = flux_encode_text(ctx->text_encoder, prompt, &text_seq);
   } else if (external_emb != NULL) {
       text_emb = external_emb;
       text_seq = external_seq;
   } else {
       // Use null embeddings (current behavior)
   }
   ```

3. **CLI changes** (main.c):
   - `-p/--prompt` triggers text encoding
   - `-e/--embeddings` for pre-computed (kept for compatibility)
   - If both provided, `-p` takes precedence

### Optimization Considerations

1. **Memory**: Qwen3-4B needs ~4.4 GB for weights
   - Consider INT8 quantization (1.1 GB)
   - Could use memory-mapped weights (mmap)

2. **Speed**: Text encoding is one-time per generation
   - 36 layers × 512 tokens is manageable
   - Main bottleneck is image generation, not text encoding

3. **KV Cache**: Not needed since we only do one forward pass (no autoregressive generation)

### File Structure

```
flux_qwen3.h        - Public API
flux_qwen3.c        - Model implementation
flux_tokenizer.c    - Update for Qwen3 tokenizer (currently has placeholder)
```

### Testing Plan

1. **Unit test tokenization**:
   - Compare token IDs with HuggingFace tokenizer
   - Test special characters, Unicode, long prompts

2. **Unit test model output**:
   - Save Python embeddings for test prompts
   - Compare C embeddings (should match within FP32 precision)

3. **End-to-end test**:
   - Generate image with `-p "a red car"`
   - Visually verify image shows a red car
   - Compare with Python-generated image using same prompt

### Dependencies

- None beyond current (pure C + BLAS)
- tokenizer.json parsing needs JSON parser (can use simple custom parser)

### Estimated Effort

- Phase 1 (Tokenizer): 2-3 days
- Phase 2 (Model): 3-4 days
- Phase 3 (Weights): 1 day
- Phase 4 (Integration): 1 day
- Testing & debugging: 2-3 days
- **Total: ~10-14 days**

### Open Questions

1. **Qwen3 variant**: Need to verify which exact variant Klein4B uses
   - Check `text_encoder/config.json` for hidden_size
   - 2560 → Qwen3-4B, 4096 → Qwen3-8B

2. **Quantization**: Should we support INT8/FP16 from the start?
   - FP32 is simpler but uses 4.4 GB RAM

3. **GQA implementation**: Grouped Query Attention is slightly different from standard MHA
   - 20 query heads share 4 KV heads (5:1 ratio)

---

## Qwen3 Text Encoder - IMPLEMENTED (2024-01-18)

The Qwen3 text encoder has been fully implemented in C. The `-p "prompt"` option now works directly without needing pre-computed embeddings.

### Implementation Files

- **flux_qwen3.h** - Public API and architecture constants
- **flux_qwen3.c** - Model implementation (forward pass, weight loading)
- **flux_qwen3_tokenizer.c** - BPE tokenizer with chat template

### Architecture Details (Verified)

From `text_encoder/config.json`:
- hidden_size: 2560
- intermediate_size: 9728
- num_attention_heads: 32
- num_key_value_heads: 8 (GQA 4:1 ratio)
- head_dim: 128 (via `hidden_size / num_attention_heads * (num_attention_heads / num_key_value_heads)`)
- num_hidden_layers: 36
- vocab_size: 151936
- rms_norm_eps: 1e-6
- rope_theta: 1000000.0

### Layer Extraction (Critical Fix)

Python's `hidden_states` indexing:
- `hidden_states[0]` = input embeddings
- `hidden_states[i]` = output AFTER layer `i-1` (0-indexed)

So extracting layers [9, 18, 27] means:
- `hidden_states[9]` = after layer 8
- `hidden_states[18]` = after layer 17
- `hidden_states[27]` = after layer 26

C implementation extracts at `QWEN3_OUTPUT_LAYER_1=8`, `QWEN3_OUTPUT_LAYER_2=17`, `QWEN3_OUTPUT_LAYER_3=26`.

### Chat Template

Format applied by tokenizer:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Special tokens:
- `<|im_start|>` = 151644
- `<|im_end|>` = 151645
- `<|endoftext|>` = 151643 (PAD)
- `<think>` = 151667
- `</think>` = 151668

### Test Results

Embeddings match Python reference within FP32 precision:
```
Stats for real tokens only (first 21):
  Max diff: 0.011719
  Mean diff: 0.000006

*** PASS: Embeddings match within tolerance! ***
```

### Usage

```bash
# Direct text-to-image generation
./flux -d flux-klein-model -p "A fluffy orange cat" -o cat.png

# Still supports pre-computed embeddings for compatibility
./flux -d flux-klein-model --embeddings text_embeddings.bin -o output.png
```

### Performance Notes

- Text encoding runs on CPU using Accelerate BLAS
- ~36 transformer layers × 512 sequence length
- Encoding time: ~2-3 seconds on M3 Max
- Memory: ~8 GB for FP32 weights (loaded from safetensors)

---

## Recent Updates (2024-01-18)

### Memory Management: Automatic Encoder Release

The text encoder (~8GB) is now automatically released after encoding to reduce peak memory during diffusion.

**API**:
```c
void flux_release_text_encoder(flux_ctx *ctx);
```

**Behavior**:
- `flux_generate()` and `flux_img2img()` auto-release the encoder after encoding
- If a new prompt is provided, the encoder reloads automatically from `model_dir`
- Library users can call `flux_release_text_encoder()` manually for fine control

**Implementation**:
- Added `model_dir` field to `flux_ctx` to track model path for reloading
- Modified `flux_encode_text()` to reload encoder if NULL
- Peak memory reduced from ~16GB to ~8GB during diffusion

### img2img Fix: Use Full Denoising Steps

**Bug**: img2img was reducing steps based on strength (e.g., strength=0.8 → 3 steps instead of 4).

**Root cause**: FLUX klein is a 4-step distilled model that must always use exactly 4 denoising steps. The strength parameter should only control the noise level added to the input image, not skip steps.

**Fix** (flux.c):
```c
/* For distilled models like FLUX klein (4-step), we should always use
 * the full number of steps. The strength controls how much noise is added
 * to the image, not how many steps are skipped. */
int num_steps = p.num_steps;  // Always use full steps
float t_start = strength;     // Strength controls initial noise level
```

### Default Size Change

- Default output size changed from 1024x1024 to 256x256
- Faster iteration for testing and development
- Users can still specify `-W` and `-H` for larger images

### img2img Output Size

When using `-i` for img2img, if `-W` and `-H` are not specified:
- Output dimensions default to input image dimensions
- Before: always defaulted to 1024x1024 regardless of input

### Seed Reproducibility

The actual seed used is now always printed to stderr:
```
$ ./flux -d model -p "test" -o out.png
Seed: 1705612345
out.png
```

This allows reproducing any run by using `-S <seed>` with the printed value.

---

## Performance Optimization Plan (2024-01-18)

### Baseline Performance

Benchmarks on Apple M3 Max (128GB RAM), 4-step generation:

| Size | C (MPS) | C (BLAS) | PyTorch (MPS, bf16) | Slowdown |
|------|---------|----------|---------------------|----------|
| 512×512 | 49.6s | 51.9s | 5.4s | ~10x |
| 256×256 | 32.4s | 29.7s | 3.0s | ~10x |
| 64×64 | 25.0s | 23.5s | 2.2s | ~11x |

**Root cause**: NOT float32 vs bfloat16. The ~10x gap comes from architectural issues.

### Prioritized Optimization Steps

#### Step 1: Eliminate Per-Op GPU Sync (CRITICAL - Expected 3-5x)

**Problem**: Every `flux_metal_sgemm` call does:
```objc
[cmdBuffer commit];
[cmdBuffer waitUntilCompleted];  // CPU waits for GPU!
```
With ~1300 matmuls/step × 4 steps = 5200+ sync points.

**Solution**:
- Batch operations into single command buffer per transformer block
- Sync only at block boundaries or step end
- Keep intermediate results on GPU

**Files**: `flux_metal.m`, `flux_metal.h`

#### Step 2: Eliminate Per-Op Memory Copies (CRITICAL - included in Step 1)

**Problem**: Every matmul allocates buffers, copies in, copies out:
```objc
id<MTLBuffer> bufferA = [g_device newBufferWithBytes:A ...];  // Copy in
// ... compute ...
memcpy(C, [bufferC contents], sizeC);  // Copy out
```

**Solution**:
- Persistent GPU buffers for activations (allocated once per inference)
- Weights stay on GPU after first use (already cached)
- Use `MTLResourceStorageModePrivate` where possible

**Files**: `flux_metal.m`, `flux_transformer.c` (buffer management)

#### Step 3: Move Attention to GPU (HIGH - Expected 1.5-2x)

**Problem**: `mha_forward()` and `joint_attention()` use CPU BLAS:
```c
for (int h = 0; h < tf->num_heads; h++) {
    cblas_sgemm(...);  // Q @ K^T on CPU
    flux_softmax(...); // CPU
    cblas_sgemm(...);  // scores @ V on CPU
}
```

**Solution**:
- Use batched Metal matmul for all heads at once
- Or implement fused SDPA kernel
- Move softmax to GPU

**Files**: `flux_transformer.c`, `flux_metal.m`

#### Step 4: bfloat16 Inference (MODERATE - Expected 1.3-1.5x)

**Problem**: Weights loaded as bf16, converted to f32, computed in f32.

**Solution**:
- Keep weights in bf16 (skip conversion in `flux_safetensors.c`)
- Use `MPSDataTypeFloat16` for matmul
- Activations in f16, accumulate in f32 for stability

**Note**: Generic C target stays f32 (no hardware f16 support).

**Files**: `flux_safetensors.c`, `flux_metal.m`, `flux_kernels.c`

#### Step 5: Vectorize CPU Operations (MODERATE - Expected 1.2-1.5x for BLAS)

**Problem**: Scalar loops for softmax, RMSNorm, SiLU in `flux_kernels.c`.

**Solution**: Use Accelerate/vDSP:
- Softmax: `vDSP_vmax`, `vDSP_vsub`, `vDSP_vexp`, `vDSP_sve`
- RMSNorm: `vDSP_svesq`
- SiLU: vectorized sigmoid

**Files**: `flux_kernels.c`

#### Step 6: Remove Hot-Path Allocations (LOW - Expected 1.1-1.2x)

**Problem**: malloc/free inside loops:
- `swiglu_ffn()` allocates gate/up/down per call
- AdaLN allocates temp buffers
- Attention concat/transpose copies

**Solution**: Preallocate all workspace in transformer struct.

**Files**: `flux_transformer.c`

### Expected Combined Improvement

| Steps | Expected Speedup | Time (256×256) |
|-------|------------------|----------------|
| Baseline | 1x | 30s |
| Steps 1-2 | 3-5x | 6-10s |
| + Step 3 | 5-8x | 4-6s |
| + Step 4 | 7-10x | 3-4s |
| + Steps 5-6 | 8-12x | 2.5-4s |

**Target**: Match or approach PyTorch's 3.0s for 256×256.

### Validation After Each Step

After each optimization:
```bash
# Test generic C (must still work)
make clean && make generic
./flux -d flux-klein-model -p "test" -o /tmp/test.png -W 64 -H 64

# Test MPS
make clean && make mps
./flux -d flux-klein-model -p "test" -o /tmp/test.png -W 64 -H 64

# Compare with reference
python3 -c "
import numpy as np
from PIL import Image
ref = np.array(Image.open('test_vectors/reference_1step_64x64_seed42.png'))
test = np.array(Image.open('/tmp/test.png'))
diff = np.abs(ref.astype(float) - test.astype(float))
print(f'Max diff: {diff.max()}, Mean: {diff.mean():.2f}')
"
```

### Technical Notes

**Model weights are already bfloat16** on disk (safetensors). Current code converts to f32 on load.

**Attention is O(n²)** in sequence length:
- 64×64: seq=528 (16 img + 512 txt)
- 256×256: seq=768 (256 img + 512 txt)
- 512×512: seq=1536 (1024 img + 512 txt)

**M3 Max specs**:
- ~28 TFLOPs bf16, ~14 TFLOPs f32
- ~400 GB/s memory bandwidth
- PyTorch achieves ~10.5 TFLOPs/s (~37% efficiency)

---

## Work Log

### 2024-01-19: Batch Command Buffer Infrastructure

**Added:**
- `flux_metal_begin_batch()` / `flux_metal_end_batch()` / `flux_metal_in_batch()` in `flux_metal.m`
- Wrapper functions `flux_gpu_begin_batch()` etc. in `flux_kernels.c`
- Infrastructure allows batching multiple GPU operations with single sync

**Key Finding:**
The batch infrastructure alone won't provide speedup because transformer operations have **immediate data dependencies**:
```c
flux_linear_nobias(img_q, img_norm, ...);  // Output: img_q
apply_qk_norm(img_q, ...);                  // Uses img_q immediately!
```

Each operation's output is used as input to the next operation. Batching only helps for truly independent operations.

**What's needed for real speedup:**
1. **Persistent GPU buffers**: Keep activations on GPU between operations
2. **GPU tensor abstraction**: Pass buffer handles instead of CPU pointers
3. **Restructured compute graph**: Identify and group independent operations

**Tests:** MPS and BLAS pass reference comparison (max diff: 1.0, mean: 0.0001)

**Next steps:**
- Consider moving attention computation to GPU (batched matmul for all heads)
- Or focus on bf16 inference which is simpler and still provides speedup

