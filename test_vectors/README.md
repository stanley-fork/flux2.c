# Test Vectors

Reference images for verifying that optimizations don't break inference.

## reference_1step_64x64_seed42.png

Generated with:
```bash
./flux --dir flux-klein-model --embeddings text_embeddings_official.bin --seed 42 --steps 1 --output test_vectors/reference_1step_64x64_seed42.png --height 64 --width 64
```

Parameters:
- Model: flux-klein-model (safetensors format)
- Text embeddings: text_embeddings_official.bin (512 tokens, 7680 dim)
- Seed: 42
- Steps: 1
- Size: 64x64 pixels (4x4 latent)
- Timestep schedule: official Flux2 schedule with mu-based time shift

The text embeddings were generated from the prompt "a cute orange tabby cat sitting on a windowsill, photorealistic" using the official Flux2 text encoder.

## Verification

After any optimization, regenerate the image and compare:
```bash
./flux --dir flux-klein-model --embeddings text_embeddings_official.bin --seed 42 --steps 1 --output test_output.png --height 64 --width 64

# Compare (should be identical or very close)
python3 -c "
import numpy as np
from PIL import Image
ref = np.array(Image.open('test_vectors/reference_1step_64x64_seed42.png'))
test = np.array(Image.open('test_output.png'))
diff = np.abs(ref.astype(float) - test.astype(float))
print(f'Max diff: {diff.max()}, Mean diff: {diff.mean():.4f}')
if diff.max() < 2:
    print('PASS: Images match')
else:
    print('FAIL: Images differ significantly')
"
```
