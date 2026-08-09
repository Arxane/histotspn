from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from conditioned_classifier import DualBagClassifier
from conch_runner import load_conch_model
from get_data import HistoTSPNDataLoader
from get_patches import PatchRecord, QuPathPatchExtractor
from tspn import TSPN


@dataclass(frozen=True)
class PairedWSISample:
	image_id: str
	label: int
	he_image: np.ndarray
	ihc_image: np.ndarray


@dataclass(frozen=True)
class FeaturePair:
	image_id: str
	label: int
	he_features: torch.Tensor
	ihc_features: torch.Tensor
	he_patch_count: int
	ihc_patch_count: int


@dataclass(frozen=True)
class PipelineConfig:
	data_root: str = "BCI_dataset"
	patch_size: int = 256
	stride: int = 256
	min_tissue_fraction: float = 0.5
	conch_batch_size: int = 32
	input_dim: int = 512
	output_size: int = 64
	n_seeds: int = 64
	fspool_k: int = 16
	d_model: int = 512
	nhead: int = 8
	n_heads: int = 8
	num_decoder_layers: int = 3
	dim_feedforward: int = 2048
	dropout: float = 0.1
	classifier_dim: int = 512
	classifier_heads: int = 8
	classifier_layers: int = 2
	classifier_hidden: int = 256
	num_classes: int = 4


class CompleteHistologyPipeline(nn.Module):
	def __init__(self, tspn: TSPN, classifier: DualBagClassifier):
		super().__init__()
		self.tspn = tspn
		self.classifier = classifier

	def forward(
		self,
		he_features: torch.Tensor,
		target_ihc_features: Optional[torch.Tensor] = None,
		return_tspn: bool = False,
	):
		synthetic_ihc = self.tspn(he_features)
		logits = self.classifier(he_features, synthetic_ihc)

		if not return_tspn:
			return logits

		output = {
			"synthetic_ihc_features": synthetic_ihc,
			"logits": logits,
		}
		if target_ihc_features is not None:
			output["target_ihc_features"] = target_ihc_features
		return output


def load_paired_wsi_samples(data_root: str, split: str) -> List[PairedWSISample]:
	loader = HistoTSPNDataLoader(data_root=data_root)
	he_images, he_labels, he_ids = loader.he_loader.load_images_and_labels(split)
	ihc_images, ihc_labels, ihc_ids = loader.ihc_loader.load_images_and_labels(split)

	he_lookup = {image_id: (image, int(label)) for image, label, image_id in zip(he_images, he_labels, he_ids)}
	ihc_lookup = {image_id: (image, int(label)) for image, label, image_id in zip(ihc_images, ihc_labels, ihc_ids)}

	paired_ids = sorted(set(he_lookup) & set(ihc_lookup))
	samples: List[PairedWSISample] = []

	for image_id in paired_ids:
		he_image, he_label = he_lookup[image_id]
		ihc_image, ihc_label = ihc_lookup[image_id]
		if he_label != ihc_label:
			continue
		samples.append(
			PairedWSISample(
				image_id=image_id,
				label=he_label,
				he_image=he_image,
				ihc_image=ihc_image,
			)
		)

	if not samples:
		raise RuntimeError(f"No paired HE/IHC samples were found for split '{split}'")

	return samples


def build_patch_extractor(config: PipelineConfig) -> QuPathPatchExtractor:
	return QuPathPatchExtractor(
		patch_size=config.patch_size,
		stride=config.stride,
		min_tissue_fraction=config.min_tissue_fraction,
		blur_kernel=5,
		morph_kernel=5,
	)


def extract_patch_records(image: np.ndarray, label: int, image_id: str, extractor: QuPathPatchExtractor) -> List[PatchRecord]:
	return extractor.extract_patches_from_image(image, label=label, source_id=image_id)


def encode_patch_records(
	records: Sequence[PatchRecord],
	model: torch.nn.Module,
	preprocess,
	device: torch.device,
	batch_size: int,
) -> torch.Tensor:
	if not records:
		raise ValueError("Cannot encode an empty patch list")

	encoded_batches: List[torch.Tensor] = []
	num_batches = (len(records) + batch_size - 1) // batch_size
	# Encode each batch of patches through CONCH feature extractor
	for start in tqdm(range(0, len(records), batch_size), total=num_batches, desc="  CONCH encoding", leave=False, unit="batch"):
		batch_records = records[start : start + batch_size]
		batch_images = [preprocess(Image.fromarray(record.patch).convert("RGB")) for record in batch_records]
		batch_tensor = torch.stack(batch_images, dim=0).to(device)

		with torch.inference_mode():
			embeddings = model.encode_image(batch_tensor, proj_contrast=False, normalize=False)

		encoded_batches.append(embeddings.detach().cpu())

	return torch.cat(encoded_batches, dim=0)


def build_feature_pairs(
	samples: Sequence[PairedWSISample],
	extractor: QuPathPatchExtractor,
	model: torch.nn.Module,
	preprocess,
	device: torch.device,
	batch_size: int,
) -> List[FeaturePair]:
	feature_pairs: List[FeaturePair] = []

	# Outer loop: patch extraction + CONCH encoding per WSI pair
	for sample in tqdm(samples, desc="[Stage] Building feature pairs (patch → CONCH)", unit="WSI"):
		he_records = extract_patch_records(sample.he_image, sample.label, f"{sample.image_id}_HE", extractor)
		ihc_records = extract_patch_records(sample.ihc_image, sample.label, f"{sample.image_id}_IHC", extractor)

		# Encode H&E patches
		tqdm.write(f"  [{sample.image_id}] H&E: {len(he_records)} patches | IHC: {len(ihc_records)} patches")
		he_features = encode_patch_records(he_records, model, preprocess, device, batch_size)
		# Encode IHC patches
		ihc_features = encode_patch_records(ihc_records, model, preprocess, device, batch_size)

		feature_pairs.append(
			FeaturePair(
				image_id=sample.image_id,
				label=sample.label,
				he_features=he_features,
				ihc_features=ihc_features,
				he_patch_count=len(he_records),
				ihc_patch_count=len(ihc_records),
			)
		)

	return feature_pairs


def build_tspn_model(config: PipelineConfig, device: torch.device) -> TSPN:
	return TSPN(
		input_dim=config.input_dim,
		output_size=config.n_seeds,
		n_seeds=config.n_seeds,
		fspool_k=config.fspool_k,
		d_model=config.d_model,
		nhead=config.nhead,
		n_heads=config.n_heads,
		num_decoder_layers=config.num_decoder_layers,
		dim_feedforward=config.dim_feedforward,
		dropout=config.dropout,
	).to(device)


def build_conditioned_classifier(config: PipelineConfig, device: torch.device) -> DualBagClassifier:
	return DualBagClassifier(
		in_dim=config.input_dim,
		dim=config.classifier_dim,
		num_heads=config.classifier_heads,
		num_layers=config.classifier_layers,
		mlp_hidden=config.classifier_hidden,
		num_classes=config.num_classes,
		dropout=config.dropout,
	).to(device)


def build_complete_pipeline(config: PipelineConfig, device: torch.device) -> CompleteHistologyPipeline:
	return CompleteHistologyPipeline(
		tspn=build_tspn_model(config, device),
		classifier=build_conditioned_classifier(config, device),
	)


def freeze_module(module: nn.Module) -> nn.Module:
	for parameter in module.parameters():
		parameter.requires_grad = False
	module.eval()
	return module


def unfreeze_module(module: nn.Module) -> nn.Module:
	for parameter in module.parameters():
		parameter.requires_grad = True
	return module


def save_feature_pairs(feature_pairs: Sequence[FeaturePair], out_path: Path) -> None:
	payload = []
	for pair in feature_pairs:
		payload.append(
			{
				"image_id": pair.image_id,
				"label": pair.label,
				"he_features": pair.he_features,
				"ihc_features": pair.ihc_features,
				"he_patch_count": pair.he_patch_count,
				"ihc_patch_count": pair.ihc_patch_count,
			}
		)
	torch.save(payload, out_path)


def save_manifest(feature_pairs: Sequence[FeaturePair], out_path: Path, extra: Optional[Dict[str, object]] = None) -> None:
	manifest = {
		"num_samples": len(feature_pairs),
		"samples": [
			{
				"image_id": pair.image_id,
				"label": pair.label,
				"he_patch_count": pair.he_patch_count,
				"ihc_patch_count": pair.ihc_patch_count,
			}
			for pair in feature_pairs
		],
	}
	if extra:
		manifest.update(extra)
	out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_state_dict(module: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
	checkpoint = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint.get("model_state_dict", checkpoint)
	module.load_state_dict(state_dict)
	return module


def load_tspn_config_from_checkpoint(checkpoint_path: Path, base_config: PipelineConfig) -> PipelineConfig:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	saved = checkpoint.get("config", {})

	# Map from train_tspn.py argparse names to PipelineConfig field names
	tspn_field_map = {
		"output_size":        "output_size",
		"n_seeds":            "n_seeds",
		"fspool_k":           "fspool_k",
		"d_model":            "d_model",
		"nhead":              "nhead",
		"n_heads":            "n_heads",
		"num_decoder_layers": "num_decoder_layers",
		"dim_feedforward":    "dim_feedforward",
		"dropout":            "dropout",
	}

	overrides = {dst: saved[src] for src, dst in tspn_field_map.items() if src in saved}

	if overrides:
		tqdm.write(f"[Config] TSPN architecture loaded from checkpoint: {overrides}")
	else:
		tqdm.write("[Config] Warning: no architecture config in checkpoint — using PipelineConfig defaults")

	base_fields = {f: getattr(base_config, f) for f in base_config.__dataclass_fields__}
	return PipelineConfig(**{**base_fields, **overrides})


@torch.no_grad()
def export_tspn_outputs(
	model: TSPN,
	feature_pairs: Sequence[FeaturePair],
	device: torch.device,
) -> Dict[str, object]:
	model.eval()
	exported: Dict[str, object] = {
		"synthetic_ihc_patches": [],
		"labels": [],
		"sample_ids": [],
	}

	# Generate synthetic IHC feature sets from H&E via trained TSPN
	for pair in tqdm(feature_pairs, desc="[Export] TSPN → synthetic IHC", unit="WSI"):
		synthetic_ihc = model(pair.he_features.to(device)).cpu()
		exported["synthetic_ihc_patches"].append(synthetic_ihc)
		exported["labels"].append(pair.label)
		exported["sample_ids"].append(pair.image_id)

	return exported


@torch.no_grad()
def export_complete_predictions(
	model: CompleteHistologyPipeline,
	feature_pairs: Sequence[FeaturePair],
	device: torch.device,
) -> Dict[str, Dict[str, object]]:
	model.eval()
	exported: Dict[str, Dict[str, object]] = {}

	# Run full pipeline (TSPN + classifier) and collect predictions per WSI
	for pair in tqdm(feature_pairs, desc="[Export] Full pipeline inference", unit="WSI"):
		outputs = model(pair.he_features.to(device), return_tspn=True)
		logits = outputs["logits"].cpu()
		synthetic_ihc = outputs["synthetic_ihc_features"].cpu()
		exported[pair.image_id] = {
			"label": pair.label,
			"logits": logits,
			"prediction": int(logits.argmax(dim=-1).item()),
			"synthetic_ihc_patches": synthetic_ihc,
			"real_ihc_patches": pair.ihc_features,
		}

	return exported


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Shared HistoTSPN pipeline utilities")
	subparsers = parser.add_subparsers(dest="command", required=True)

	prepare_parser = subparsers.add_parser("prepare", help="Load data, extract patches, and encode CONCH features")
	prepare_parser.add_argument("--data-root", type=str, default="BCI_dataset")
	prepare_parser.add_argument("--split", type=str, default="train", choices=("train", "validation", "test"))
	prepare_parser.add_argument("--output-dir", type=str, default="artifacts/pipeline")
	prepare_parser.add_argument("--patch-size", type=int, default=256)
	prepare_parser.add_argument("--stride", type=int, default=256)
	prepare_parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
	prepare_parser.add_argument("--conch-batch-size", type=int, default=32)
	prepare_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

	sanity_parser = subparsers.add_parser("sanity-check", help="Run a variable-length shape consistency check")
	sanity_parser.add_argument("--he-dim", type=int, default=512)
	sanity_parser.add_argument("--n-seeds", type=int, default=64)
	sanity_parser.add_argument("--n-classes", type=int, default=4)

	tspn_parser = subparsers.add_parser("export-tspn", help="Export synthetic IHC features from a trained TSPN")
	tspn_parser.add_argument("--checkpoint", type=str, required=True)
	tspn_parser.add_argument("--data-root", type=str, default="BCI_dataset")
	tspn_parser.add_argument("--split", type=str, default="test", choices=("train", "validation", "test"))
	tspn_parser.add_argument("--output-dir", type=str, default="artifacts/pipeline")
	tspn_parser.add_argument("--patch-size", type=int, default=256)
	tspn_parser.add_argument("--stride", type=int, default=256)
	tspn_parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
	tspn_parser.add_argument("--conch-batch-size", type=int, default=32)
	tspn_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

	complete_parser = subparsers.add_parser("export-complete", help="Run TSPN plus conditioned classifier")
	complete_parser.add_argument("--tspn-checkpoint", type=str, required=True)
	complete_parser.add_argument("--classifier-checkpoint", type=str, required=True)
	complete_parser.add_argument("--data-root", type=str, default="BCI_dataset")
	complete_parser.add_argument("--split", type=str, default="test", choices=("train", "validation", "test"))
	complete_parser.add_argument("--output-dir", type=str, default="artifacts/pipeline")
	complete_parser.add_argument("--patch-size", type=int, default=256)
	complete_parser.add_argument("--stride", type=int, default=256)
	complete_parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
	complete_parser.add_argument("--conch-batch-size", type=int, default=32)
	complete_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

	train_full_parser = subparsers.add_parser("train-full", help="Run Stage 0, Stage 1 distillation, and Stage 2 fine-tuning")
	train_full_parser.add_argument("--data-root", type=str, default="BCI_dataset")
	train_full_parser.add_argument("--output-base", type=str, default="artifacts")
	train_full_parser.add_argument("--patch-size", type=int, default=256)
	train_full_parser.add_argument("--stride", type=int, default=256)
	train_full_parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
	train_full_parser.add_argument("--conch-batch-size", type=int, default=32)
	train_full_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	train_full_parser.add_argument("--epochs-stage1", type=int, default=20, help="Epochs for Stage 1 TSPN distillation")
	train_full_parser.add_argument("--epochs-stage2", type=int, default=20, help="Epochs for Stage 0 real-modality classifier")
	train_full_parser.add_argument("--epochs-stage3", type=int, default=5, help="Epochs for Stage 2 joint fine-tuning")
	train_full_parser.add_argument("--num-decoder-layers", type=int, default=3, help="TSPN Transformer decoder layers")
	train_full_parser.add_argument("--classifier-layers", type=int, default=2, help="Conditioned classifier cross-attention layers")
	train_full_parser.add_argument("--alpha", type=float, default=1.0)
	train_full_parser.add_argument("--beta", type=float, default=0.5)
	train_full_parser.add_argument("--gamma", type=float, default=0.3)
	train_full_parser.add_argument("--delta", type=float, default=2.0)
	train_full_parser.add_argument("--eta", type=float, default=0.5)
	train_full_parser.add_argument("--temperature", type=float, default=4.0)
	train_full_parser.add_argument("--n-seeds", type=int, default=64)
	train_full_parser.add_argument("--n-heads", type=int, default=8)
	train_full_parser.add_argument("--learning-rate", type=float, default=1e-4)
	train_full_parser.add_argument("--weight-decay", type=float, default=1e-5)

	return parser.parse_args()


def sanity_check(he_dim=512, n_seeds=64, n_classes=4):
	he_patches = torch.randn(37, he_dim)
	ihc_patches = torch.randn(51, he_dim)
	tspn = TSPN(dim=he_dim, n_seeds=n_seeds)
	classifier = DualBagClassifier(
		in_dim=he_dim,
		dim=he_dim,
		num_heads=8,
		num_layers=2,
		mlp_hidden=256,
		num_classes=n_classes,
	)
	syn_ihc = tspn(he_patches)
	assert syn_ihc.shape == (n_seeds, he_dim), f"TSPN output shape wrong: {syn_ihc.shape}"
	logits = classifier(he_patches, syn_ihc)
	assert logits.shape == (n_classes,), f"Classifier output shape wrong: {logits.shape}"
	print("Sanity check passed.")


def _prepare_feature_pairs(args: argparse.Namespace) -> Tuple[PipelineConfig, List[FeaturePair]]:
	config = PipelineConfig(
		data_root=args.data_root,
		patch_size=args.patch_size,
		stride=args.stride,
		min_tissue_fraction=args.min_tissue_fraction,
		conch_batch_size=args.conch_batch_size,
	)
	device = torch.device(args.device)

	# Load CONCH model (downloads weights on first run)
	tqdm.write("[Prepare] Loading CONCH feature extractor...")
	conch_model, preprocess = load_conch_model()
	conch_model = conch_model.to(device)
	conch_model.eval()

	tqdm.write(f"[Prepare] Loading paired WSI samples from '{config.data_root}' ({args.split} split)...")
	samples = load_paired_wsi_samples(config.data_root, args.split)
	tqdm.write(f"[Prepare] Found {len(samples)} paired HE/IHC samples")
	extractor = build_patch_extractor(config)
	feature_pairs = build_feature_pairs(samples, extractor, conch_model, preprocess, device, config.conch_batch_size)

	return config, feature_pairs


def _run_prepare(args: argparse.Namespace) -> None:
	config, feature_pairs = _prepare_feature_pairs(args)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	save_feature_pairs(feature_pairs, output_dir / f"{args.split}_feature_pairs.pt")
	save_manifest(
		feature_pairs,
		output_dir / f"{args.split}_manifest.json",
		extra={"config": asdict(config), "split": args.split},
	)
	print(f"Prepared {len(feature_pairs)} paired samples for {args.split} split")


def _run_export_tspn(args: argparse.Namespace) -> None:
	config, feature_pairs = _prepare_feature_pairs(args)
	device = torch.device(args.device)

	config = load_tspn_config_from_checkpoint(Path(args.checkpoint), config)
	model = build_tspn_model(config, device)
	load_state_dict(model, Path(args.checkpoint), device)

	exported = export_tspn_outputs(model, feature_pairs, device)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	torch.save(exported, output_dir / f"{args.split}_synthetic_ihc.pt")
	print(f"Exported TSPN outputs for {len(exported['sample_ids'])} samples")


def _run_export_complete(args: argparse.Namespace) -> None:
	config, feature_pairs = _prepare_feature_pairs(args)
	device = torch.device(args.device)

	config = load_tspn_config_from_checkpoint(Path(args.tspn_checkpoint), config)
	pipeline = build_complete_pipeline(config, device)
	load_state_dict(pipeline.tspn, Path(args.tspn_checkpoint), device)
	load_state_dict(pipeline.classifier, Path(args.classifier_checkpoint), device)

	exported = export_complete_predictions(pipeline, feature_pairs, device)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	torch.save(exported, output_dir / f"{args.split}_complete_predictions.pt")
	print(f"Exported full-model predictions for {len(exported)} samples")


def _run_full_pipeline(args: argparse.Namespace) -> None:
	"""
	Orchestrate all three training stages in sequence:
	1. Stage 1: Train TSPN (train_tspn.py)
	2. Stage 2: Train classifier with frozen TSPN (train_classifier.py)
	3. Stage 3: Fine-tune full pipeline (finetune_final.py)
	"""
	base_dir = Path(args.output_base)
	base_dir.mkdir(parents=True, exist_ok=True)

	# Top-level progress bar tracking the three pipeline stages
	stage_bar = tqdm(total=3, desc="[Pipeline] Overall progress", unit="stage", position=0)

	# Stage 1: Train TSPN
	print("\n" + "=" * 80)
	print("STAGE 1/3: Training TSPN  [H&E patches → synthetic IHC feature sets]")
	print("=" * 80)
	stage1_output = base_dir / "tspn_training"
	stage1_cmd = [
		sys.executable,
		"train_tspn.py",
		"--data-root", args.data_root,
		"--output-dir", str(stage1_output),
		"--patch-size", str(args.patch_size),
		"--stride", str(args.stride),
		"--min-tissue-fraction", str(args.min_tissue_fraction),
		"--conch-batch-size", str(args.conch_batch_size),
		"--device", args.device,
		"--epochs", str(args.epochs_stage1),
		"--classifier-checkpoint", str(base_dir / "classifier_real" / "classifier_checkpoint.pt"),
		"--alpha", str(args.alpha), "--beta", str(args.beta), "--gamma", str(args.gamma),
		"--delta", str(args.delta), "--eta", str(args.eta), "--temperature", str(args.temperature),
		"--n-seeds", str(args.n_seeds), "--n-heads", str(args.n_heads),
		"--num-decoder-layers", str(args.num_decoder_layers),
		"--learning-rate", str(args.learning_rate),
		"--weight-decay", str(args.weight_decay),
	]
	result = subprocess.run(stage1_cmd)
	if result.returncode != 0:
		print(f"\n❌ Stage 1 failed with return code {result.returncode}")
		sys.exit(1)
	
	stage1_checkpoint = stage1_output / "tspn_checkpoint.pt"
	if not stage1_checkpoint.exists():
		print(f"\n❌ Stage 1 checkpoint not found: {stage1_checkpoint}")
		sys.exit(1)
	stage_bar.update(1)
	tqdm.write(f"✓ Stage 1 complete. Checkpoint: {stage1_checkpoint}")

	# Stage 2: Train classifier (TSPN frozen, only DualBagClassifier trains)
	print("\n" + "=" * 80)
	print("STAGE 2/3: Training Classifier  [frozen TSPN → cross-bag attention → HER2 score]")
	print("=" * 80)
	stage2_output = base_dir / "classifier_stage"
	stage2_cmd = [
		sys.executable,
		"train_classifier.py",
		"--data-root", args.data_root,
		"--tspn-checkpoint", str(stage1_checkpoint),
		"--output-dir", str(stage2_output),
		"--patch-size", str(args.patch_size),
		"--stride", str(args.stride),
		"--min-tissue-fraction", str(args.min_tissue_fraction),
		"--conch-batch-size", str(args.conch_batch_size),
		"--device", args.device,
		"--epochs", str(args.epochs_stage2),
		"--classifier-layers", str(args.classifier_layers),
		"--learning-rate", str(args.learning_rate),
		"--weight-decay", str(args.weight_decay),
		"--save-synthetic-features",
	]
	result = subprocess.run(stage2_cmd)
	if result.returncode != 0:
		print(f"\n❌ Stage 2 failed with return code {result.returncode}")
		sys.exit(1)
	
	stage2_checkpoint = stage2_output / "classifier_checkpoint.pt"
	if not stage2_checkpoint.exists():
		print(f"\n❌ Stage 2 checkpoint not found: {stage2_checkpoint}")
		sys.exit(1)
	stage_bar.update(1)
	tqdm.write(f"✓ Stage 2 complete. Checkpoint: {stage2_checkpoint}")

	# Stage 3: End-to-end fine-tuning of TSPN + classifier jointly
	print("\n" + "=" * 80)
	print("STAGE 3/3: Fine-tuning Full Pipeline  [joint TSPN + classifier, composite loss]")
	print("=" * 80)
	stage3_output = base_dir / "final_stage"
	stage3_cmd = [
		sys.executable,
		"finetune_final.py",
		"--data-root", args.data_root,
		"--tspn-checkpoint", str(stage1_checkpoint),
		"--classifier-checkpoint", str(stage2_checkpoint),
		"--output-dir", str(stage3_output),
		"--patch-size", str(args.patch_size),
		"--stride", str(args.stride),
		"--min-tissue-fraction", str(args.min_tissue_fraction),
		"--conch-batch-size", str(args.conch_batch_size),
		"--device", args.device,
		"--epochs", str(args.epochs_stage3),
		"--classifier-layers", str(args.classifier_layers),
		"--tspn-learning-rate", "1e-5",
		"--classifier-learning-rate", "1e-4",
		"--weight-decay", str(args.weight_decay),
		"--mmd-weight", "0.1",
	]
	result = subprocess.run(stage3_cmd)
	if result.returncode != 0:
		print(f"\n❌ Stage 3 failed with return code {result.returncode}")
		sys.exit(1)
	
	stage3_checkpoint = stage3_output / "finetuned_pipeline_checkpoint.pt"
	if not stage3_checkpoint.exists():
		print(f"\n❌ Stage 3 checkpoint not found: {stage3_checkpoint}")
		sys.exit(1)
	stage_bar.update(1)
	stage_bar.close()
	tqdm.write(f"✓ Stage 3 complete. Checkpoint: {stage3_checkpoint}")

	print("\n" + "=" * 80)
	print("✓ FULL PIPELINE TRAINING COMPLETE")
	print("=" * 80)
	print(f"Stage 1 (TSPN):       {stage1_checkpoint}")
	print(f"Stage 2 (Classifier): {stage2_checkpoint}")
	print(f"Stage 3 (Fine-tune):  {stage3_checkpoint}")
	print("=" * 80 + "\n")


def _run_distillation_protocol(args: argparse.Namespace) -> None:
	"""Run Stage 0 real-modality training, Stage 1 distillation, and Stage 2 fine-tuning."""
	base_dir = Path(args.output_base)
	base_dir.mkdir(parents=True, exist_ok=True)

	stage0_output = base_dir / "classifier_real"
	stage0_cmd = [
		sys.executable, "train_classifier_real.py",
		"--data-root", args.data_root, "--output-dir", str(stage0_output),
		"--patch-size", str(args.patch_size), "--stride", str(args.stride),
		"--min-tissue-fraction", str(args.min_tissue_fraction),
		"--conch-batch-size", str(args.conch_batch_size), "--device", args.device,
		"--epochs", str(args.epochs_stage2), "--classifier-layers", str(args.classifier_layers),
		"--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay),
	]
	print("STAGE 0/3: Training classifier on real HE + real IHC")
	result = subprocess.run(stage0_cmd)
	if result.returncode != 0:
		raise SystemExit(f"Stage 0 failed with return code {result.returncode}")
	stage0_checkpoint = stage0_output / "classifier_checkpoint.pt"

	stage1_output = base_dir / "tspn_distill"
	stage1_cmd = [
		sys.executable, "train_tspn_distill.py",
		"--data-root", args.data_root, "--classifier-checkpoint", str(stage0_checkpoint),
		"--output-dir", str(stage1_output), "--patch-size", str(args.patch_size),
		"--stride", str(args.stride), "--min-tissue-fraction", str(args.min_tissue_fraction),
		"--conch-batch-size", str(args.conch_batch_size), "--device", args.device,
		"--epochs", str(args.epochs_stage1), "--learning-rate", str(args.learning_rate),
		"--weight-decay", str(args.weight_decay), "--alpha", str(args.alpha),
		"--delta", str(args.delta), "--eta", str(args.eta), "--temperature", str(args.temperature),
		"--n-seeds", str(args.n_seeds), "--n-heads", str(args.n_heads),
		"--num-decoder-layers", str(args.num_decoder_layers),
		"--beta", str(args.beta), "--gamma", str(args.gamma),
	]
	print("STAGE 1/3: Training TSPN with fidelity + distillation + task losses")
	result = subprocess.run(stage1_cmd)
	if result.returncode != 0:
		raise SystemExit(f"Stage 1 failed with return code {result.returncode}")
	stage1_checkpoint = stage1_output / "tspn_checkpoint.pt"

	stage2_output = base_dir / "final_stage"
	stage2_cmd = [
		sys.executable, "finetune_final.py", "--data-root", args.data_root,
		"--tspn-checkpoint", str(stage1_checkpoint),
		"--classifier-checkpoint", str(stage0_checkpoint), "--output-dir", str(stage2_output),
		"--patch-size", str(args.patch_size), "--stride", str(args.stride),
		"--min-tissue-fraction", str(args.min_tissue_fraction),
		"--conch-batch-size", str(args.conch_batch_size), "--device", args.device,
		"--epochs", str(args.epochs_stage3), "--classifier-layers", str(args.classifier_layers),
		"--tspn-learning-rate", "1e-5", "--classifier-learning-rate", "1e-4",
		"--weight-decay", str(args.weight_decay), "--classification-weight", "1.0",
		"--mmd-weight", "0.1",
	]
	print("STAGE 2/3: Jointly fine-tuning TSPN and classifier")
	result = subprocess.run(stage2_cmd)
	if result.returncode != 0:
		raise SystemExit(f"Stage 2 failed with return code {result.returncode}")
	stage2_checkpoint = stage2_output / "finetuned_pipeline_checkpoint.pt"

	print("\nTraining protocol complete")
	print(f"Stage 0 classifier: {stage0_checkpoint}")
	print(f"Stage 1 TSPN:       {stage1_checkpoint}")
	print(f"Stage 2 pipeline:   {stage2_checkpoint}")


def main() -> None:
	args = parse_args()

	if args.command == "sanity-check":
		sanity_check(args.he_dim, args.n_seeds, args.n_classes)
	elif args.command == "prepare":
		_run_prepare(args)
	elif args.command == "export-tspn":
		_run_export_tspn(args)
	elif args.command == "export-complete":
		_run_export_complete(args)
	elif args.command == "train-full":
		_run_distillation_protocol(args)
	else:
		raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
	main()