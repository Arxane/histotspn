from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from biomed_runner import load_biomedclip_model

from main import PipelineConfig, build_complete_pipeline, build_patch_extractor, build_feature_pairs, load_paired_wsi_samples, load_tspn_config_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate fine-tuned HistoTSPN pipeline under:\n" "  1. HE only\n" "  2. HE + real IHC\n" "  3. HE + synthetic IHC")
    )

    parser.add_argument("--data-root", type=str, default="BCI_dataset")

    parser.add_argument("--split", type=str, default="validation", choices=("train", "validation", "test"))

    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        required=True,
        help=("Fine-tuned pipeline checkpoint, e.g. " "artifacts/final_stage/finetuned_pipeline_checkpoint.pt"),
    )

    parser.add_argument("--patch-size", type=int, default=128)

    parser.add_argument("--stride", type=int, default=128)

    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)

    parser.add_argument("--conch-batch-size", type=int, default=32)

    parser.add_argument("--classifier-layers", type=int, default=2)

    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--output", type=str, default=("artifacts/final_stage/" "condition_results.json"))

    return parser.parse_args()


def build_eval_feature_pairs(args: argparse.Namespace):
    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
        classifier_layers=args.classifier_layers,
    )

    device = torch.device(args.device)

    print(f"[Data] Loading BiomedCLIP feature extractor...")

    feature_model, preprocess, _ = load_biomedclip_model()

    feature_model = feature_model.to(device).eval()

    print(f"[Data] Loading {args.split} split...")

    samples = load_paired_wsi_samples(config.data_root, args.split)

    print(f"[Data] Found {len(samples)} paired samples")

    # IMPORTANT:
    # for the current experiment.

    cache_root = Path("feature_cache") / "883bd10e" / args.split

    print(f"[Data] Feature cache: {cache_root}")

    feature_pairs = build_feature_pairs(
        samples, build_patch_extractor(config), feature_model, preprocess, device, config.conch_batch_size, cache_root=cache_root, normalize_features=True
    )

    print(f"[Data] Loaded {len(feature_pairs)} feature pairs")

    return config, feature_pairs


def make_confusion_matrix(targets, predictions, num_classes):
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for target, prediction in zip(targets, predictions):
        cm[target][prediction] += 1

    return cm


def calculate_per_class_metrics(cm):

    num_classes = len(cm)

    metrics = []

    for c in range(num_classes):

        tp = cm[c][c]

        fp = sum(cm[r][c] for r in range(num_classes) if r != c)

        fn = sum(cm[c][r] for r in range(num_classes) if r != c)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0

        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        metrics.append({"class": c, "precision": precision, "recall": recall})

    return metrics


@torch.no_grad()
def evaluate_condition(model, feature_pairs, device, condition, num_classes=4):

    model.eval()

    predictions = []
    targets = []

    for pair in tqdm(feature_pairs, desc=condition, leave=False):

        he = pair.he_features.to(device)

        real_ihc = pair.ihc_features.to(device)

        #
        # NOTE:
        # genuine HE-only forward path.
        #
        # Therefore this is an IHC-MASKED ablation:
        # IHC features are replaced by zeros.

        if condition == "he_only":

            ihc = torch.zeros_like(real_ihc)

        elif condition == "he_real_ihc":

            ihc = real_ihc

        elif condition == "he_synthetic_ihc":

            ihc = model.tspn(he)

            ihc = torch.nn.functional.normalize(ihc, dim=-1)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        logits = model.classifier(he, ihc)

        prediction = logits.argmax(dim=-1).item()

        predictions.append(prediction)
        targets.append(pair.label)

    accuracy = sum(p == t for p, t in zip(predictions, targets)) / max(len(targets), 1)

    cm = make_confusion_matrix(targets, predictions, num_classes)

    per_class = calculate_per_class_metrics(cm)

    return {"accuracy": accuracy, "confusion_matrix": cm, "per_class": per_class, "num_samples": len(targets)}


def main():

    args = parse_args()

    device = torch.device(args.device)

    checkpoint_path = Path(args.finetuned_checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist:\n" f"{checkpoint_path}")

    print("=" * 80)
    print("HistoTSPN CONDITION EVALUATION")
    print("=" * 80)

    print(f"Checkpoint : {checkpoint_path}")

    print(f"Split      : {args.split}")

    print(f"Device     : {device}")

    print()

    config, feature_pairs = build_eval_feature_pairs(args)

    config = load_tspn_config_from_checkpoint(checkpoint_path, config)

    print("[Model] Building pipeline...")

    model = build_complete_pipeline(config, device)

    print("[Model] Loading fine-tuned TSPN...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "tspn_state_dict" not in checkpoint:
        raise KeyError("Fine-tuned checkpoint does not contain " "'tspn_state_dict'.")

    if "classifier_state_dict" not in checkpoint:
        raise KeyError("Fine-tuned checkpoint does not contain " "'classifier_state_dict'.")

    model.tspn.load_state_dict(checkpoint["tspn_state_dict"])

    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])

    model.tspn.eval()
    model.classifier.eval()

    print("[Model] Fine-tuned pipeline loaded.")

    print()

    conditions = [("he_only", "HE only"), ("he_real_ihc", "HE + real IHC"), ("he_synthetic_ihc", "HE + synthetic IHC")]

    results = {}

    for key, name in conditions:

        print("=" * 80)
        print(f"Evaluating: {name}")
        print("=" * 80)

        result = evaluate_condition(model=model, feature_pairs=feature_pairs, device=device, condition=key, num_classes=config.num_classes)

        results[key] = result

        print(f"{name}: " f"{result['accuracy'] * 100:.2f}%")

        print()
        print("Confusion matrix:")

        for row in result["confusion_matrix"]:
            print(row)

        print()

        print("Per-class:")

        for item in result["per_class"]:

            print(f"  Class {item['class']}: " f"precision={item['precision']:.4f}, " f"recall={item['recall']:.4f}")

        print()

    he_only = results["he_only"]["accuracy"]

    real_ihc = results["he_real_ihc"]["accuracy"]

    synthetic_ihc = results["he_synthetic_ihc"]["accuracy"]

    comparison = {
        "he_only": he_only,
        "he_real_ihc": real_ihc,
        "he_synthetic_ihc": synthetic_ihc,
        "real_ihc_gain_over_he_only": real_ihc - he_only,
        "synthetic_ihc_gain_over_he_only": synthetic_ihc - he_only,
        "synthetic_vs_real_difference": synthetic_ihc - real_ihc,
        "synthetic_vs_real_absolute_difference": abs(synthetic_ihc - real_ihc),
    }

    output = {
        "num_samples": len(feature_pairs),
        "split": args.split,
        "num_classes": config.num_classes,
        "conditions": {
            "he_only": {"description": ("HE with IHC branch " "masked by zero features"), "accuracy": results["he_only"]["accuracy"]},
            "he_real_ihc": {"description": "HE + real IHC", "accuracy": results["he_real_ihc"]["accuracy"]},
            "he_synthetic_ihc": {"description": "HE + TSPN synthetic IHC", "accuracy": results["he_synthetic_ihc"]["accuracy"]},
        },
        "detailed_results": results,
        "comparison": comparison,
        "checkpoint": str(checkpoint_path),
        "feature_cache": str(Path("feature_cache") / "883bd10e" / args.split),
        "configuration": {"patch_size": args.patch_size, "stride": args.stride, "min_tissue_fraction": args.min_tissue_fraction, "device": args.device},
    }

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=" * 80)
    print("FINAL ACCURACY COMPARISON")
    print("=" * 80)

    print(f"HE only:             " f"{he_only * 100:.2f}%")

    print(f"HE + real IHC:       " f"{real_ihc * 100:.2f}%")

    print(f"HE + synthetic IHC:  " f"{synthetic_ihc * 100:.2f}%")

    print()

    print(f"Real IHC vs HE:      " f"{(real_ihc - he_only) * 100:+.2f} pp")

    print(f"Synthetic IHC vs HE:" f" {(synthetic_ihc - he_only) * 100:+.2f} pp")

    print(f"Synthetic vs real:   " f"{(synthetic_ihc - real_ihc) * 100:+.2f} pp")

    print()

    print(f"Results saved to:\n" f"{output_path}")


if __name__ == "__main__":
    main()
