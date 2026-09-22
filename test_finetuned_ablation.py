from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from main import PipelineConfig, build_complete_pipeline, build_feature_pairs, build_patch_extractor, load_paired_wsi_samples
from biomed_runner import load_biomedclip_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation evaluation of the fine-tuned HistoTSPN pipeline.")

    parser.add_argument("--data-root", type=str, default="BCI_dataset")

    parser.add_argument("--split", type=str, default="validation", choices=("train", "validation", "test"))

    parser.add_argument("--checkpoint", type=str, default="artifacts/final_stage/finetuned_pipeline_checkpoint.pt")

    parser.add_argument("--output-dir", type=str, default="artifacts/ablation")

    parser.add_argument("--patch-size", type=int, default=128)

    parser.add_argument("--stride", type=int, default=128)

    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)

    parser.add_argument("--conch-batch-size", type=int, default=32)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_finetuned_pipeline(checkpoint_path: Path, base_config: PipelineConfig, device: torch.device):
    print(f"[Checkpoint] Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "tspn_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain 'tspn_state_dict'. " "Expected a Stage-3 finetuned pipeline checkpoint.")

    if "classifier_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain 'classifier_state_dict'. " "Expected a Stage-3 finetuned pipeline checkpoint.")

    saved_config = checkpoint.get("config", {})

    config_fields = {field: getattr(base_config, field) for field in base_config.__dataclass_fields__}

    architecture_fields = (
        "input_dim",
        "output_size",
        "n_seeds",
        "fspool_k",
        "d_model",
        "nhead",
        "n_heads",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
        "classifier_dim",
        "classifier_heads",
        "classifier_layers",
        "classifier_hidden",
        "num_classes",
        "lmf_rank",
        "lmf_output_dim",
    )

    for field in architecture_fields:
        if field in saved_config:
            config_fields[field] = saved_config[field]

    config = PipelineConfig(**config_fields)

    print("[Checkpoint] Architecture:")
    print(f"  input_dim           = {config.input_dim}")
    print(f"  n_seeds             = {config.n_seeds}")
    print(f"  d_model             = {config.d_model}")
    print(f"  decoder_layers      = {config.num_decoder_layers}")
    print(f"  classifier_dim      = {config.classifier_dim}")
    print(f"  classifier_layers   = {config.classifier_layers}")
    print(f"  lmf_rank            = {config.lmf_rank}")
    print(f"  lmf_output_dim      = {config.lmf_output_dim}")

    model = build_complete_pipeline(config, device)

    # IMPORTANT:
    # tspn_state_dict and classifier_state_dict.
    #
    # Do NOT pass the entire checkpoint to TSPN.

    model.tspn.load_state_dict(checkpoint["tspn_state_dict"], strict=True)

    model.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)

    model.eval()

    print(f"[Checkpoint] Best epoch: " f"{checkpoint.get('best_epoch', 'unknown')}")

    print(f"[Checkpoint] Best eval loss: " f"{checkpoint.get('best_eval_loss', 'unknown')}")

    return model, config


def load_feature_pairs(args: argparse.Namespace):
    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
    )

    device = torch.device(args.device)

    # IMPORTANT:
    #
    # Your current normalized cache is:
    #
    # feature_cache/883bd10e/validation
    #
    # Use it directly rather than generating another hash.

    cache_root = Path("feature_cache") / "883bd10e" / args.split

    print(f"[Features] Cache: {cache_root}")

    if not cache_root.exists():
        raise FileNotFoundError(f"Feature cache does not exist:\n{cache_root}\n\n" "This script intentionally does not silently " "re-encode features.")

    print("[Features] Loading BiomedCLIP...")

    feature_model, preprocess, _ = load_biomedclip_model()
    feature_model = feature_model.to(device).eval()

    samples = load_paired_wsi_samples(config.data_root, args.split)

    print(f"[Features] Found {len(samples)} paired " f"HE/IHC WSIs.")

    extractor = build_patch_extractor(config)

    pairs = build_feature_pairs(samples, extractor, feature_model, preprocess, device, config.conch_batch_size, cache_root=cache_root, normalize_features=True)

    if not pairs:
        raise RuntimeError("No feature pairs were loaded.")

    he_norms = torch.cat([pair.he_features.norm(dim=-1) for pair in pairs])

    ihc_norms = torch.cat([pair.ihc_features.norm(dim=-1) for pair in pairs])

    print("\n[Features] Normalization check")
    print(f"  HE   mean norm = {he_norms.mean().item():.6f}")
    print(f"  HE   min/max   = " f"{he_norms.min().item():.6f} / " f"{he_norms.max().item():.6f}")
    print(f"  IHC  mean norm = {ihc_norms.mean().item():.6f}")
    print(f"  IHC  min/max   = " f"{ihc_norms.min().item():.6f} / " f"{ihc_norms.max().item():.6f}")

    return config, pairs


def confusion_matrix(labels: List[int], predictions: List[int], num_classes: int):
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for y, p in zip(labels, predictions):
        cm[y][p] += 1

    return cm


def per_class_metrics(cm: List[List[int]]):
    num_classes = len(cm)
    results = []

    for c in range(num_classes):

        tp = cm[c][c]

        fp = sum(cm[r][c] for r in range(num_classes) if r != c)

        fn = sum(cm[c][k] for k in range(num_classes) if k != c)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0

        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        results.append({"class": c, "precision": precision, "recall": recall})

    return results


@torch.no_grad()
def predict_condition(model, he, ihc):
    logits = model.classifier(he, ihc)

    prediction = int(logits.argmax(dim=-1).item())

    return logits, prediction


@torch.no_grad()
def evaluate(args):

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    _, feature_pairs = load_feature_pairs(args)

    base_config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
    )

    model, config = load_finetuned_pipeline(Path(args.checkpoint), base_config, device)

    num_classes = config.num_classes

    conditions = {
        "he_only": {"description": "HE + zero IHC"},
        "he_real_ihc": {"description": "HE + real IHC"},
        "he_synthetic_ihc": {"description": "HE + TSPN synthetic IHC"},
        "he_random_ihc": {"description": "HE + random normalized IHC"},
    }

    outputs = {condition: {"labels": [], "predictions": [], "logits": []} for condition in conditions}

    print("\n[Evaluation] Running ablation...")

    for pair in tqdm(feature_pairs, desc="Evaluating", unit="WSI"):

        he = pair.he_features.to(device)
        real_ihc = pair.ihc_features.to(device)

        synthetic_ihc = model.tspn(he)

        # IMPORTANT:
        # Real cached IHC is normalized.
        # Therefore synthetic IHC must also be normalized.
        synthetic_ihc = F.normalize(synthetic_ihc, dim=-1)

        #
        # IHC branch, but replace their features with zeros.

        zero_ihc = torch.zeros_like(real_ihc)

        #
        # Same shape as real IHC, independently sampled and
        # L2-normalized.

        random_ihc = torch.randn_like(real_ihc)

        random_ihc = F.normalize(random_ihc, dim=-1)

        condition_inputs = {"he_only": zero_ihc, "he_real_ihc": real_ihc, "he_synthetic_ihc": synthetic_ihc, "he_random_ihc": random_ihc}

        for condition, ihc in condition_inputs.items():

            logits, prediction = predict_condition(model, he, ihc)

            outputs[condition]["labels"].append(pair.label)

            outputs[condition]["predictions"].append(prediction)

            outputs[condition]["logits"].append(logits.detach().cpu())

    detailed_results = {}

    for condition, data in outputs.items():

        labels = data["labels"]
        predictions = data["predictions"]

        correct = sum(int(y == p) for y, p in zip(labels, predictions))

        accuracy = correct / max(len(labels), 1)

        cm = confusion_matrix(labels, predictions, num_classes)

        detailed_results[condition] = {"accuracy": accuracy, "confusion_matrix": cm, "per_class": per_class_metrics(cm), "num_samples": len(labels)}

    def stack_logits(condition):
        return torch.stack(outputs[condition]["logits"])

    logits_zero = stack_logits("he_only")
    logits_real = stack_logits("he_real_ihc")
    logits_synthetic = stack_logits("he_synthetic_ihc")
    logits_random = stack_logits("he_random_ihc")

    real_vs_zero = torch.norm(logits_real - logits_zero, dim=-1)

    synthetic_vs_zero = torch.norm(logits_synthetic - logits_zero, dim=-1)

    real_vs_synthetic = torch.norm(logits_real - logits_synthetic, dim=-1)

    random_vs_zero = torch.norm(logits_random - logits_zero, dim=-1)

    logit_diagnostics = {
        "real_vs_zero": {"mean": float(real_vs_zero.mean()), "median": float(real_vs_zero.median()), "std": float(real_vs_zero.std())},
        "synthetic_vs_zero": {"mean": float(synthetic_vs_zero.mean()), "median": float(synthetic_vs_zero.median()), "std": float(synthetic_vs_zero.std())},
        "real_vs_synthetic": {"mean": float(real_vs_synthetic.mean()), "median": float(real_vs_synthetic.median()), "std": float(real_vs_synthetic.std())},
        "random_vs_zero": {"mean": float(random_vs_zero.mean()), "median": float(random_vs_zero.median()), "std": float(random_vs_zero.std())},
    }

    pred_zero = outputs["he_only"]["predictions"]
    pred_real = outputs["he_real_ihc"]["predictions"]
    pred_synthetic = outputs["he_synthetic_ihc"]["predictions"]
    pred_random = outputs["he_random_ihc"]["predictions"]

    def change_rate(a, b):
        return sum(int(x != y) for x, y in zip(a, b)) / max(len(a), 1)

    prediction_changes = {
        "real_vs_zero": change_rate(pred_real, pred_zero),
        "synthetic_vs_zero": change_rate(pred_synthetic, pred_zero),
        "real_vs_synthetic": change_rate(pred_real, pred_synthetic),
        "random_vs_zero": change_rate(pred_random, pred_zero),
    }

    accuracy = {condition: detailed_results[condition]["accuracy"] for condition in detailed_results}

    accuracy_comparison = {
        "he_only": accuracy["he_only"],
        "he_real_ihc": accuracy["he_real_ihc"],
        "he_synthetic_ihc": accuracy["he_synthetic_ihc"],
        "he_random_ihc": accuracy["he_random_ihc"],
        "real_gain_over_he_only": (accuracy["he_real_ihc"] - accuracy["he_only"]),
        "synthetic_gain_over_he_only": (accuracy["he_synthetic_ihc"] - accuracy["he_only"]),
        "random_gain_over_he_only": (accuracy["he_random_ihc"] - accuracy["he_only"]),
        "synthetic_vs_real_difference": (accuracy["he_synthetic_ihc"] - accuracy["he_real_ihc"]),
    }

    results = {
        "num_samples": len(feature_pairs),
        "split": args.split,
        "num_classes": num_classes,
        "conditions": {condition: {"description": data["description"], "accuracy": accuracy[condition]} for condition, data in conditions.items()},
        "accuracy_comparison": accuracy_comparison,
        "detailed_results": detailed_results,
        "logit_diagnostics": logit_diagnostics,
        "prediction_change_rates": prediction_changes,
        "checkpoint": args.checkpoint,
        "feature_cache": str(Path("feature_cache") / "883bd10e" / args.split),
        "configuration": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "min_tissue_fraction": args.min_tissue_fraction,
            "device": args.device,
            "seed": args.seed,
        },
    }

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.split}_finetuned_ablation.json"

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n")
    print("=" * 72)
    print("FINETUNED PIPELINE ABLATION")
    print("=" * 72)

    print(f"HE only             : " f"{accuracy['he_only'] * 100:.2f}%")

    print(f"HE + real IHC       : " f"{accuracy['he_real_ihc'] * 100:.2f}%")

    print(f"HE + synthetic IHC  : " f"{accuracy['he_synthetic_ihc'] * 100:.2f}%")

    print(f"HE + random IHC    : " f"{accuracy['he_random_ihc'] * 100:.2f}%")

    print("-" * 72)

    print("Real IHC vs HE-only " f"logit distance      : " f"{logit_diagnostics['real_vs_zero']['mean']:.6f}")

    print("Synthetic vs HE-only " f"logit distance      : " f"{logit_diagnostics['synthetic_vs_zero']['mean']:.6f}")

    print("Real vs synthetic " f"logit distance       : " f"{logit_diagnostics['real_vs_synthetic']['mean']:.6f}")

    print("Random vs HE-only " f"logit distance       : " f"{logit_diagnostics['random_vs_zero']['mean']:.6f}")

    print("-" * 72)

    print("Prediction changes " f"(real vs zero)      : " f"{prediction_changes['real_vs_zero'] * 100:.2f}%")

    print("Prediction changes " f"(synthetic vs zero) : " f"{prediction_changes['synthetic_vs_zero'] * 100:.2f}%")

    print("Prediction changes " f"(real vs synthetic) : " f"{prediction_changes['real_vs_synthetic'] * 100:.2f}%")

    print("Prediction changes " f"(random vs zero)    : " f"{prediction_changes['random_vs_zero'] * 100:.2f}%")

    print("=" * 72)

    print(f"\nSaved results to:\n{output_path}")


if __name__ == "__main__":
    evaluate(parse_args())
