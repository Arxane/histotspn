#!/usr/bin/env python
"""
smoke_test_biomedclip.py

Quick smoke test for the BiomedCLIP feature extractor integration.
Tests model loading, image preprocessing, and feature extraction without
requiring full training data or GPU memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Test 1: Module imports
print("[1/5] Testing module imports...")
try:
    from biomed_runner import load_biomedclip_model, encode_biomedclip_image
    from main import (
        build_patch_extractor,
        PipelineConfig,
        encode_patch_records,
    )
    from get_patches import PatchRecord
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model loading
print("\n[2/5] Testing BiomedCLIP model loading...")
try:
    model, preprocess = load_biomedclip_model()
    print(f"✓ Model loaded: {type(model)}")
    print(f"✓ Preprocess loaded: {type(preprocess)}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Test 3: Create a synthetic image and test preprocessing
print("\n[3/5] Testing image preprocessing...")
try:
    # Create a synthetic RGB image (256x256, typical patch size)
    synthetic_image = Image.new("RGB", (256, 256), color=(128, 64, 200))
    
    # Preprocess the image
    preprocessed = preprocess(synthetic_image)
    print(f"✓ Synthetic image created: {synthetic_image.size}")
    print(f"✓ Preprocessed shape: {preprocessed.shape}")
    assert preprocessed.dim() == 3, f"Expected 3D tensor, got {preprocessed.dim()}D"
except Exception as e:
    print(f"✗ Image preprocessing failed: {e}")
    sys.exit(1)

# Test 4: Encode a batch of synthetic patches
print("\n[4/5] Testing batch image feature encoding...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    model = model.to(device).eval()
    
    # Create a batch of synthetic patches
    batch_size = 4
    batch_images = [Image.new("RGB", (256, 256), color=(128 + i*20, 64, 200)) for i in range(batch_size)]
    batch_tensors = [preprocess(img).unsqueeze(0) for img in batch_images]
    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
    
    print(f"✓ Created batch: {batch_tensor.shape}")
    
    # Encode the batch
    with torch.inference_mode():
        features = model.encode_image(batch_tensor, normalize=False)
    
    print(f"✓ Encoded features shape: {features.shape}")
    assert features.shape[0] == batch_size, f"Expected batch size {batch_size}, got {features.shape[0]}"
    assert features.shape[1] == 512, f"Expected feature dim 512, got {features.shape[1]}"
    print(f"✓ Feature dimension is 512 (CONCH/BiomedCLIP standard)")
except Exception as e:
    print(f"✗ Batch encoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Integrate with patch extraction pipeline
print("\n[5/5] Testing integration with patch extraction pipeline...")
try:
    config = PipelineConfig()
    extractor = build_patch_extractor(config)
    
    # Create a synthetic WSI-sized numpy array (1024x1024, 3 channels)
    synthetic_wsi = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    
    print(f"✓ Synthetic WSI shape: {synthetic_wsi.shape}")
    
    # Extract patches
    patch_records = extractor.extract_patches_from_image(
        synthetic_wsi,
        label=0,
        source_id="smoke_test_wsi"
    )
    print(f"✓ Extracted {len(patch_records)} patches")
    
    if patch_records:
        # Encode the patches through the model
        model = model.to(device).eval()
        encoded = encode_patch_records(
            patch_records,
            model,
            preprocess,
            device,
            batch_size=4
        )
        print(f"✓ Encoded patches shape: {encoded.shape}")
        assert encoded.shape[0] == len(patch_records), "Mismatch in number of encoded patches"
        assert encoded.shape[1] == 512, f"Expected feature dim 512, got {encoded.shape[1]}"
        print(f"✓ Integration successful: {len(patch_records)} patches → {encoded.shape[0]} features")
    else:
        print("⚠ No patches extracted (tissue fraction may be low for synthetic image)")
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All smoke tests passed!")
print("="*60)
print("\nThe BiomedCLIP feature extractor pipeline is ready to use.")
print("Next steps:")
print("  - Train the classifier: python train_classifier_real.py")
print("  - Train TSPN: python train_tspn.py --classifier-checkpoint <path>")
print("  - Full pipeline: python main.py train-full")
