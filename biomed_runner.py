from __future__ import annotations

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

from pathlib import Path
from typing import Sequence

import dotenv
import torch
from PIL import Image
from tqdm import tqdm
import hashlib
import json


def _cache_key(image_paths: Sequence[Path | str], model_name: str) -> str:
    """Stable hash of the file list + model name."""
    paths_str = json.dumps([str(p) for p in sorted(image_paths)])
    return hashlib.md5((paths_str + model_name).encode()).hexdigest()[:16]


def encode_biomedclip_images_cached(
    image_paths: Sequence[Path | str], cache_dir: Path | str = "feature_cache", batch_size: int = 32, device: str | None = None, model_name: str = MODEL_NAME
) -> torch.Tensor:
    """
    Encode images and cache results to disk.
    On subsequent calls with the same image list, loads from cache instantly.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(image_paths, model_name)
    cache_path = cache_dir / f"{key}.pt"

    if cache_path.exists():
        print(f"[conch] Loading features from cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"[conch] No cache found - encoding {len(image_paths)} images ...")
    features = encode_biomedclip_images(image_paths, batch_size=batch_size, device=device)

    torch.save(features, cache_path)
    print(f"[conch] Features cached to {cache_path}")
    return features


try:
    from open_clip import create_model_from_pretrained, get_tokenizer
except ImportError:
    create_model_from_pretrained = None
    get_tokenizer = None

try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None

__all__ = ["load_biomedclip_model", "encode_biomedclip_image", "encode_biomedclip_images", "load_conch_model", "encode_conch_image", "encode_conch_images"]


_model = None
_preprocess = None
_tokenizer = None
_device = None
_feat_dim = None


def _ensure_open_clip_available():
    if create_model_from_pretrained is None or get_tokenizer is None:
        raise ImportError("open_clip is not installed. " "pip install open_clip_torch")


def _authenticate_hf():
    """
    Log in to HuggingFace Hub using the token in .env.
    HF recommends authenticating even for public models - it bypasses
    anonymous rate limits and gives priority bandwidth on the CDN.
    """
    hf_token = dotenv.get_key(".env", "HF_TOKEN")
    if hf_token and hf_login is not None:
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
            print("[conch] HuggingFace authenticated - rate limits lifted.")
        except Exception as e:
            print(f"[conch] HF login failed (continuing anyway): {e}")
    elif hf_login is None:
        print("[conch] huggingface_hub not installed - skipping auth. " "pip install huggingface_hub for better download speeds.")
    else:
        print("[conch] No HF_TOKEN found in .env - anonymous access.")


class _FeatureExtractorWrapper(torch.nn.Module):
    """Wraps OpenCLIP model to accept CONCH-compatible encode_image kwargs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def encode_image(self, images, proj_contrast=False, normalize=True):
        del proj_contrast, normalize
        return self.model.encode_image(images)

    def forward(self, images, proj_contrast=False, normalize=True):
        return self.encode_image(images, proj_contrast=proj_contrast, normalize=normalize)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[conch] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[conch] No GPU found - running on CPU (will be slow).")
    return dev


def load_biomedclip_model(model_name: str = MODEL_NAME, device: str | None = None):
    global _model, _preprocess, _tokenizer, _device, _feat_dim

    if _model is not None:
        return _model, _preprocess, _device

    _ensure_open_clip_available()
    _authenticate_hf()

    _device = _resolve_device(device)

    print(f"[conch] Loading {model_name} ...")
    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)

    _model = _FeatureExtractorWrapper(model).to(_device).eval()
    _preprocess = preprocess
    _tokenizer = tokenizer

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224, device=_device)
        _feat_dim = _model.encode_image(dummy).shape[-1]
    print(f"[conch] Model ready. Feature dim = {_feat_dim}, device = {_device}")

    return _model, _preprocess, _device


def encode_biomedclip_image(image_path: Path | str, device: str | None = None) -> torch.Tensor:
    model, preprocess, dev = load_biomedclip_model(device=device)
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(dev)
    with torch.inference_mode():
        emb = model.encode_image(image_tensor, normalize=True)
    return emb.cpu()


def encode_biomedclip_images(image_paths: Sequence[Path | str], batch_size: int = 32, device: str | None = None) -> torch.Tensor:
    """
    Encode a list of image paths in batches.

    batch_size=32 is a safe default for a 16 GB GPU.
    Increase to 64 if you have >=24 GB VRAM; decrease if you hit OOM.
    """
    model, preprocess, dev = load_biomedclip_model(device=device)

    if not image_paths:
        return torch.empty((0, _feat_dim))

    all_embeddings: list[torch.Tensor] = []
    batch_tensors: list[torch.Tensor] = []

    for i, image_path in enumerate(tqdm(image_paths, desc="Encoding images")):
        image = Image.open(image_path).convert("RGB")
        batch_tensors.append(preprocess(image))

        flush = (len(batch_tensors) == batch_size) or (i == len(image_paths) - 1)
        if flush:
            batch = torch.stack(batch_tensors).to(dev)
            with torch.inference_mode():
                embs = model.encode_image(batch, normalize=True)
            all_embeddings.append(embs.cpu())
            batch_tensors = []

    return torch.cat(all_embeddings, dim=0)


def load_conch_model(device: str | None = None):
    return load_biomedclip_model(device=device)


def encode_conch_image(image_path: Path | str, device: str | None = None):
    return encode_biomedclip_image(image_path, device=device)


def encode_conch_images(image_paths: Sequence[Path | str], batch_size: int = 32, device: str | None = None):
    return encode_biomedclip_images(image_paths, batch_size=batch_size, device=device)


if __name__ == "__main__":
    sample_image = Path("BCI_dataset/IHC/train/02946_train_0.png")
    emb = encode_biomedclip_image(sample_image)
    print(f"Single image embedding shape: {emb.shape}")

    sample_images = [Path("BCI_dataset/IHC/train/02946_train_0.png"), Path("BCI_dataset/HE/train/02946_train_0.png")]
    batch_embs = encode_biomedclip_images(sample_images, batch_size=2)
    print(f"Batch embeddings shape: {batch_embs.shape}")
