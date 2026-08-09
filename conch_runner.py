from pathlib import Path

import torch
import numpy as np
from PIL import Image
import dotenv
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained

__all__ = ["load_conch_model", "encode_conch_image", "encode_conch_images"]

hf_access_token = dotenv.get_key(".env", "HF_TOKEN")

_model = None
_preprocess = None


def load_conch_model():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=hf_access_token,
        )
    return _model, _preprocess


def encode_conch_image(image_path):
    model, preprocess = load_conch_model()
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        image_embs = model.encode_image(image, proj_contrast=False, normalize=False)

    return image_embs


def encode_conch_images(image_paths):
    model, preprocess = load_conch_model()
    embeddings = []

    for image_path in tqdm(image_paths, desc="Encoding CONCH images"):
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0)

        with torch.inference_mode():
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)

        embeddings.append(image_embs)

    return torch.cat(embeddings, dim=0)


if __name__ == "__main__":
    sample_image = Path("BCI_dataset/IHC/train/02946_train_0.png")
    embeddings = encode_conch_image(sample_image)
    print(f"Single image embedding shape: {embeddings.shape}")
    
    sample_images = [
        Path("BCI_dataset/IHC/train/02946_train_0.png"),
        Path("BCI_dataset/HE/train/02946_train_0.png"),
    ]
    batch_embeddings = encode_conch_images(sample_images)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
