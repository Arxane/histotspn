from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from biomed_runner import load_biomedclip_model

from main import PipelineConfig, build_complete_pipeline, build_feature_pairs, build_patch_extractor, load_paired_wsi_samples, load_tspn_config_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Evaluate the fine-tuned HE -> synthetic-IHC pipeline " "against real IHC."))

    parser.add_argument("--data-root", type=str, default="BCI_dataset")

    parser.add_argument("--split", type=str, default="validation", choices=("train", "validation", "test"))

    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        required=True,
        help=("Checkpoint produced by finetune_final.py, containing " "tspn_state_dict and classifier_state_dict."),
    )

    parser.add_argument("--output-file", type=str, default="artifacts/real_vs_synthetic/results.json")

    parser.add_argument("--patch-size", type=int, default=128)

    parser.add_argument("--stride", type=int, default=128)

    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)

    parser.add_argument("--conch-batch-size", type=int, default=32)

    parser.add_argument("--classifier-layers", type=int, default=2)

    parser.add_argument(
        "--cache-hash", type=str, default="883bd10e", help=("Feature cache version used during training. " "Your normalized cache is 883bd10e.")
    )

    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    return parser.parse_args()


def build_eval_pairs(args: argparse.Namespace, device: torch.device):
    """
    Load the evaluation features from the existing normalized
    feature cache.

    No new BiomedCLIP features should be generated when the
    cache is complete.
    """

    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
        classifier_layers=args.classifier_layers,
    )

    cache_root = Path("feature_cache") / args.cache_hash / args.split

    print(f"[Data] Feature cache: {cache_root}")

    if not cache_root.exists():
        raise FileNotFoundError(
            f"\nFeature cache does not exist:\n" f"  {cache_root}\n\n" f"Available caches:\n" f"  find feature_cache -maxdepth 3 -type d | sort"
        )

    samples = load_paired_wsi_samples(config.data_root, args.split)

    # This is only needed by build_feature_pairs().
    # Existing cache entries will be loaded instead of encoded.
    print("[Data] Loading BiomedCLIP interface " "(cache hits should skip encoding)...")

    feature_model, preprocess, _ = load_biomedclip_model()

    feature_model = feature_model.to(device).eval()

    extractor = build_patch_extractor(config)

    pairs = build_feature_pairs(samples, extractor, feature_model, preprocess, device, config.conch_batch_size, cache_root=cache_root)

    return config, pairs


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:

    device = torch.device(args.device)

    print("=" * 80)
    print("FINE-TUNED REAL IHC vs SYNTHETIC IHC EVALUATION")
    print("=" * 80)

    config, pairs = build_eval_pairs(args, device)

    print(f"\nEvaluation samples: {len(pairs)}")

    checkpoint_path = Path(args.finetuned_checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found:\n" f"  {checkpoint_path}")

    print(f"\n[Model] Loading checkpoint:\n" f"        {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "tspn_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain " "'tspn_state_dict'.")

    if "classifier_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain " "'classifier_state_dict'.")

    config = load_tspn_config_from_checkpoint(checkpoint_path, config)

    model = build_complete_pipeline(config, device)

    model.tspn.load_state_dict(checkpoint["tspn_state_dict"])

    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])

    model.tspn.eval()
    model.classifier.eval()

    print("[Model] Loaded fine-tuned TSPN " "+ classifier.")

    if "best_epoch" in checkpoint:
        print(f"[Model] Best epoch: " f"{checkpoint['best_epoch']}")

    if "best_eval_loss" in checkpoint:
        print(f"[Model] Best eval loss: " f"{checkpoint['best_eval_loss']:.6f}")

    labels = []

    real_predictions = []
    synthetic_predictions = []

    real_correct = 0
    synthetic_correct = 0

    print("\nRunning evaluation...")

    for pair in tqdm(pairs, desc="Evaluation"):

        he = pair.he_features.to(device)

        real_ihc = pair.ihc_features.to(device)

        label = int(pair.label)

        #
        # Fine-tuned classifier:
        #
        # Real HE + Real IHC

        real_logits = model.classifier(he, real_ihc)

        real_prediction = int(real_logits.argmax(dim=-1).item())

        #
        # Fine-tuned TSPN + fine-tuned classifier:
        #
        # Real HE -> Synthetic IHC
        # Real HE + Synthetic IHC

        synthetic_ihc = model.tspn(he)

        # IMPORTANT:
        # Your finetune_final.py normalizes the TSPN output
        # before passing it to the classifier.
        synthetic_ihc = torch.nn.functional.normalize(synthetic_ihc, dim=-1)

        synthetic_logits = model.classifier(he, synthetic_ihc)

        synthetic_prediction = int(synthetic_logits.argmax(dim=-1).item())

        labels.append(label)

        real_predictions.append(real_prediction)

        synthetic_predictions.append(synthetic_prediction)

        real_correct += int(real_prediction == label)

        synthetic_correct += int(synthetic_prediction == label)

    num_samples = len(labels)

    if num_samples == 0:
        raise RuntimeError("No evaluation samples found.")

    real_accuracy = real_correct / num_samples

    synthetic_accuracy = synthetic_correct / num_samples

    absolute_drop = real_accuracy - synthetic_accuracy

    relative_drop = absolute_drop / real_accuracy if real_accuracy > 0 else 0.0

    num_classes = max(max(labels), max(real_predictions), max(synthetic_predictions)) + 1

    real_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    synthetic_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for label, real_prediction, synthetic_prediction in zip(labels, real_predictions, synthetic_predictions):

        real_cm[label, real_prediction] += 1

        synthetic_cm[label, synthetic_prediction] += 1

    per_class = []

    for class_id in range(num_classes):
        real_tp = int(real_cm[class_id, class_id].item())

        real_actual = int(real_cm[class_id, :].sum().item())

        real_predicted = int(real_cm[:, class_id].sum().item())

        real_recall = real_tp / real_actual if real_actual > 0 else 0.0

        real_precision = real_tp / real_predicted if real_predicted > 0 else 0.0

        synthetic_tp = int(synthetic_cm[class_id, class_id].item())

        synthetic_actual = int(synthetic_cm[class_id, :].sum().item())

        synthetic_predicted = int(synthetic_cm[:, class_id].sum().item())

        synthetic_recall = synthetic_tp / synthetic_actual if synthetic_actual > 0 else 0.0

        synthetic_precision = synthetic_tp / synthetic_predicted if synthetic_predicted > 0 else 0.0

        per_class.append(
            {
                "class": class_id,
                "real_precision": real_precision,
                "real_recall": real_recall,
                "synthetic_precision": synthetic_precision,
                "synthetic_recall": synthetic_recall,
                "precision_drop": real_precision - synthetic_precision,
                "recall_drop": real_recall - synthetic_recall,
            }
        )

    results = {
        "num_samples": num_samples,
        "split": args.split,
        "num_classes": num_classes,
        "conditions": {
            "real_he_real_ihc": {"he": "real", "ihc": "real", "model": "fine_tuned_pipeline"},
            "real_he_synthetic_ihc": {"he": "real", "ihc": "synthetic", "model": "fine_tuned_pipeline"},
        },
        "accuracy": {
            "real_he_real_ihc": real_accuracy,
            "real_he_synthetic_ihc": synthetic_accuracy,
            "absolute_drop": absolute_drop,
            "absolute_drop_percentage_points": absolute_drop * 100.0,
            "relative_drop": relative_drop,
            "relative_drop_percent": relative_drop * 100.0,
        },
        "confusion_matrix": {"real_he_real_ihc": real_cm.tolist(), "real_he_synthetic_ihc": synthetic_cm.tolist()},
        "per_class": per_class,
        "checkpoints": {"finetuned_pipeline": str(checkpoint_path)},
        "feature_cache": str(Path("feature_cache") / args.cache_hash / args.split),
        "configuration": {"patch_size": args.patch_size, "stride": args.stride, "min_tissue_fraction": args.min_tissue_fraction, "device": args.device},
        "loss_weights": checkpoint.get("loss_weights", {}),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_eval_loss": checkpoint.get("best_eval_loss"),
    }

    output_file = Path(args.output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Condition':40s} Accuracy")

    print("-" * 65)

    print(f"{'Real HE + Real IHC':40s}" f"{real_accuracy * 100:.2f}%")

    print(f"{'Real HE + Synthetic IHC':40s}" f"{synthetic_accuracy * 100:.2f}%")

    print("-" * 65)

    print(f"Absolute accuracy drop: " f"{absolute_drop * 100:.2f} percentage points")

    print(f"Relative accuracy drop: " f"{relative_drop * 100:.2f}%")

    print("\nReal HE + Real IHC confusion matrix:")
    print(real_cm)

    print("\nReal HE + Synthetic IHC confusion matrix:")
    print(synthetic_cm)

    print("\nPer-class recall:")
    print("-" * 65)

    for item in per_class:
        print(f"Class {item['class']}: " f"real={item['real_recall']:.4f} | " f"synthetic={item['synthetic_recall']:.4f} | " f"drop={item['recall_drop']:+.4f}")

    print("\nResults saved to:")
    print(f"  {output_file}")

    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
