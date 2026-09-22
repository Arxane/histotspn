#!/usr/bin/env python

"""
smoke_test_biomedclip.py

Quick smoke test for the BiomedCLIP feature extractor integration.

Tests:
1. Module imports
2. BiomedCLIP model loading
3. Image preprocessing
4. Batch feature extraction
5. Integration with the patch extraction pipeline
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch
from PIL import Image

print("[1/5] Testing module imports...")

try:
    from biomed_runner import load_biomedclip_model, encode_biomedclip_image

    from main import PipelineConfig, build_patch_extractor, encode_patch_records

    from get_patches import PatchRecord

    print("[OK] All imports successful")

except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


print("\n[2/5] Testing BiomedCLIP model loading...")

try:
    model, preprocess, model_device = load_biomedclip_model()

    print(f"[OK] Model loaded: {type(model)}")
    print(f"[OK] Preprocess loaded: {type(preprocess)}")
    print(f"[OK] Model device reported by loader: {model_device}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    print(f"[OK] Model moved to: {device}")

except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)


print("\n[3/5] Testing image preprocessing...")

try:
    synthetic_image = Image.new("RGB", (256, 256), color=(128, 64, 200))

    print(f"[OK] Synthetic image created: " f"{synthetic_image.size}")

    preprocessed = preprocess(synthetic_image)

    print(f"[OK] Preprocessed shape: " f"{tuple(preprocessed.shape)}")

    assert preprocessed.dim() == 3, f"Expected 3D tensor, " f"got {preprocessed.dim()}D"

    assert preprocessed.shape[0] == 3, f"Expected 3 channels, " f"got {preprocessed.shape[0]}"

    print("[OK] Image preprocessing successful")

except Exception as e:
    print(f"[ERROR] Image preprocessing failed: {e}")
    traceback.print_exc()
    sys.exit(1)


print("\n[4/5] Testing batch image feature encoding...")

try:
    batch_size = 4

    batch_images = [Image.new("RGB", (256, 256), color=(128 + i * 20, 64, 200)) for i in range(batch_size)]

    batch_tensors = [preprocess(img).unsqueeze(0) for img in batch_images]

    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

    print(f"[OK] Created batch: " f"{tuple(batch_tensor.shape)}")

    with torch.inference_mode():
        features = model.encode_image(batch_tensor, normalize=False)

    print(f"[OK] Encoded features shape: " f"{tuple(features.shape)}")

    assert features.shape[0] == batch_size, f"Expected batch size {batch_size}, " f"got {features.shape[0]}"

    assert features.shape[1] == 512, f"Expected feature dimension 512, " f"got {features.shape[1]}"

    print("[OK] Feature dimension is 512 (BiomedCLIP)")

    assert torch.isfinite(features).all(), "Feature tensor contains NaN or Inf values"

    print("[OK] Features contain no NaN/Inf values")

except Exception as e:
    print(f"[ERROR] Batch encoding failed: {e}")
    traceback.print_exc()
    sys.exit(1)


print("\n[5/5] Testing integration with " "patch extraction pipeline...")

try:
    config = PipelineConfig()

    extractor = build_patch_extractor(config)

    synthetic_wsi = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

    print(f"[OK] Synthetic WSI shape: " f"{synthetic_wsi.shape}")

    patch_records = extractor.extract_patches_from_image(synthetic_wsi, label=0, source_id="smoke_test_wsi")

    print(f"[OK] Extracted {len(patch_records)} patches")

    if patch_records:

        assert all(isinstance(record, PatchRecord) for record in patch_records), "Unexpected patch record type"

        print("[OK] Patch record types are valid")

        model = model.to(device).eval()

        encoded = encode_patch_records(patch_records, model, preprocess, device, batch_size=4)

        print(f"[OK] Encoded patches shape: " f"{tuple(encoded.shape)}")

        assert encoded.shape[0] == len(patch_records), "Mismatch in number of encoded patches: " f"{encoded.shape[0]} vs {len(patch_records)}"

        assert encoded.shape[1] == 512, f"Expected feature dimension 512, " f"got {encoded.shape[1]}"

        assert torch.isfinite(encoded).all(), "Encoded patch features contain NaN or Inf values"

        print(f"[OK] Integration successful: " f"{len(patch_records)} patches -> " f"{encoded.shape[0]} features")

    else:
        print("[WARN] No patches extracted. " "This can happen if the synthetic image does not " "satisfy the configured tissue-fraction threshold.")

except Exception as e:
    print(f"[ERROR] Integration test failed: {e}")
    traceback.print_exc()
    sys.exit(1)


print("\n" + "=" * 60)
print("[OK] All smoke tests passed!")
print("=" * 60)

print("\nThe BiomedCLIP feature extractor pipeline " "is ready to use.")

print("\nNext steps:")
print("  - Train classifier:" " python train_classifier_real.py")
print("  - Train TSPN:" " python train_tspn.py " "--classifier-checkpoint <path>")
print("  - Full pipeline:" " python main.py train-full")
