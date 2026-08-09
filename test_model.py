from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from main import (
	PipelineConfig,
	build_complete_pipeline,
	build_feature_pairs,
	build_patch_extractor,
	load_paired_wsi_samples,
)
from conch_runner import load_conch_model


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate trained HistoTSPN models on the test split")
	parser.add_argument("--data-root", type=str, default="BCI_dataset")
	parser.add_argument("--final-checkpoint", type=str, default=None)
	parser.add_argument("--tspn-checkpoint", type=str, default=None)
	parser.add_argument("--classifier-checkpoint", type=str, default=None)
	parser.add_argument("--output-dir", type=str, default="validation_outputs")
	parser.add_argument("--patch-size", type=int, default=256)
	parser.add_argument("--stride", type=int, default=256)
	parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
	parser.add_argument("--conch-batch-size", type=int, default=32)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	return parser.parse_args()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
	for path in paths:
		if path.exists():
			return path
	return None


def project_path(path: str) -> Path:
	return Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path


def select_checkpoints(args: argparse.Namespace) -> Tuple[Optional[Path], Path, Path]:
	final_checkpoint = project_path(args.final_checkpoint) if args.final_checkpoint else first_existing(
		project_path(path)
		for path in (
			"artifacts/final_stage/finetuned_pipeline_checkpoint.pt",
		)
	)
	if final_checkpoint:
		return final_checkpoint, Path(), Path()

	tspn_checkpoint = project_path(args.tspn_checkpoint) if args.tspn_checkpoint else first_existing(
		project_path(path)
		for path in (
			"artifacts/tspn_distill/tspn_checkpoint.pt",
		)
	)
	classifier_checkpoint = project_path(args.classifier_checkpoint) if args.classifier_checkpoint else first_existing(
		project_path(path)
		for path in (
			"artifacts/classifier_real/classifier_checkpoint.pt",
		)
	)
	if not tspn_checkpoint or not classifier_checkpoint:
		raise FileNotFoundError(
			"Could not find trained checkpoints. Provide --final-checkpoint or both "
			"--tspn-checkpoint and --classifier-checkpoint."
		)
	return None, tspn_checkpoint, classifier_checkpoint


def checkpoint_config(checkpoint_path: Path, base_config: PipelineConfig) -> PipelineConfig:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	saved_config = checkpoint.get("config", {})
	valid_fields = {field.name for field in fields(PipelineConfig)}
	overrides = {key: value for key, value in saved_config.items() if key in valid_fields}
	return replace(base_config, **overrides)


def load_pipeline(
	args: argparse.Namespace,
	device: torch.device,
	final_checkpoint: Optional[Path],
	tspn_checkpoint: Path,
	classifier_checkpoint: Path,
) -> Tuple[torch.nn.Module, PipelineConfig]:
	base_config = PipelineConfig(
		data_root=args.data_root,
		patch_size=args.patch_size,
		stride=args.stride,
		min_tissue_fraction=args.min_tissue_fraction,
		conch_batch_size=args.conch_batch_size,
	)
	config_checkpoint = final_checkpoint or tspn_checkpoint
	config = checkpoint_config(config_checkpoint, base_config)
	if not final_checkpoint:
		config = checkpoint_config(classifier_checkpoint, config)
	config = replace(
		config,
		data_root=args.data_root,
		patch_size=args.patch_size,
		stride=args.stride,
		min_tissue_fraction=args.min_tissue_fraction,
		conch_batch_size=args.conch_batch_size,
	)
	pipeline = build_complete_pipeline(config, device)

	if final_checkpoint:
		checkpoint = torch.load(final_checkpoint, map_location=device)
		pipeline.tspn.load_state_dict(checkpoint["tspn_state_dict"])
		pipeline.classifier.load_state_dict(checkpoint["classifier_state_dict"])
	else:
		tspn_state = torch.load(tspn_checkpoint, map_location=device)
		classifier_state = torch.load(classifier_checkpoint, map_location=device)
		pipeline.tspn.load_state_dict(tspn_state.get("model_state_dict", tspn_state))
		pipeline.classifier.load_state_dict(classifier_state.get("model_state_dict", classifier_state))
	pipeline.eval()
	return pipeline, config


@torch.no_grad()
def evaluate(pipeline, feature_pairs, device: torch.device):
	correct = 0
	total = 0
	confusion = torch.zeros((4, 4), dtype=torch.long)
	predictions: Dict[str, Dict[str, object]] = {}

	for pair in tqdm(feature_pairs, desc="Evaluating test set", unit="WSI"):
		outputs = pipeline(pair.he_features.to(device), return_tspn=True)
		logits = outputs["logits"]
		synthetic = outputs["synthetic_ihc_features"]
		prediction = int(logits.argmax(dim=-1).item())
		label = int(pair.label)
		correct += int(prediction == label)
		total += 1
		confusion[label, prediction] += 1

		synthetic_pool = synthetic.mean(dim=0)
		target_pool = pair.ihc_features.to(device).mean(dim=0)
		predictions[pair.image_id] = {
			"label": label,
			"prediction": prediction,
			"probabilities": torch.softmax(logits, dim=-1).cpu().tolist(),
			"reconstruction_mse": float(F.mse_loss(synthetic_pool, target_pool).item()),
			"reconstruction_cosine": float(F.cosine_similarity(synthetic_pool, target_pool, dim=0).item()),
			"he_patch_count": pair.he_patch_count,
			"ihc_patch_count": pair.ihc_patch_count,
		}

	accuracy = correct / max(total, 1)
	per_class = []
	for class_index in range(confusion.shape[0]):
		true_positive = confusion[class_index, class_index].item()
		actual = confusion[class_index].sum().item()
		predicted = confusion[:, class_index].sum().item()
		precision = true_positive / predicted if predicted else 0.0
		recall = true_positive / actual if actual else 0.0
		per_class.append({"class": class_index, "precision": precision, "recall": recall})

	return {
		"num_samples": total,
		"accuracy": accuracy,
		"confusion_matrix": confusion.tolist(),
		"per_class": per_class,
		"mean_reconstruction_mse": sum(item["reconstruction_mse"] for item in predictions.values()) / max(total, 1),
		"mean_reconstruction_cosine": sum(item["reconstruction_cosine"] for item in predictions.values()) / max(total, 1),
	}, predictions


def main() -> None:
	args = parse_args()
	final_checkpoint, tspn_checkpoint, classifier_checkpoint = select_checkpoints(args)
	device = torch.device(args.device)
	checkpoint_label = final_checkpoint or Path(f"{tspn_checkpoint} + {classifier_checkpoint}")
	print(f"Using checkpoint(s): {checkpoint_label}")

	pipeline, config = load_pipeline(args, device, final_checkpoint, tspn_checkpoint, classifier_checkpoint)
	conch_model, preprocess = load_conch_model()
	conch_model = conch_model.to(device).eval()
	samples = load_paired_wsi_samples(config.data_root, "test")
	feature_pairs = build_feature_pairs(
		samples,
		build_patch_extractor(config),
		conch_model,
		preprocess,
		device,
		config.conch_batch_size,
	)
	metrics, predictions = evaluate(pipeline, feature_pairs, device)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
	torch.save(predictions, output_dir / "test_predictions.pt")
	print(json.dumps(metrics, indent=2))
	print(f"Saved metrics to {output_dir / 'test_metrics.json'}")
	print(f"Saved predictions to {output_dir / 'test_predictions.pt'}")


if __name__ == "__main__":
	main()