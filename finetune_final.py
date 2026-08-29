from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from biomed_runner import load_biomedclip_model
from main import (
    PipelineConfig,
    build_complete_pipeline,
    build_feature_pairs,
    build_patch_extractor,
    load_paired_wsi_samples,
    load_state_dict,
    load_tspn_config_from_checkpoint,
    unfreeze_module,
)
from tspn import {mmd_loss, swd_loss, moment_matching_loss, tspn_joint_loss}
import hashlib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly fine-tune seed-based TSPN and ABMIL classifier.")
    parser.add_argument("--data-root", type=str, default="BCI_dataset")
    parser.add_argument("--train-split", type=str, default="train", choices=("train", "validation", "test"))
    parser.add_argument("--eval-split", type=str, default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--tspn-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/final_stage")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
    parser.add_argument("--conch-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--classifier-layers", type=int, default=2)
    parser.add_argument("--tspn-learning-rate", type=float, default=1e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--mmd-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--swd-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--moment-weight",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def build_feature_pairs_for_split(args, split):
    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
        classifier_layers=args.classifier_layers,
    )
    device = torch.device(args.device)
    feature_model, preprocess, _ = load_biomedclip_model()
    feature_model = feature_model.to(device).eval()
    samples = load_paired_wsi_samples(config.data_root, split)
    cfg_hash   = hashlib.md5(
        f"{args.patch_size}_{args.stride}_{args.min_tissue_fraction}".encode()
    ).hexdigest()[:8]
    cache_root = Path("feature_cache") / cfg_hash / split
    pairs = build_feature_pairs(
        samples, build_patch_extractor(config), feature_model, preprocess, device, config.conch_batch_size, cache_root=cache_root,
    )
    return config, pairs


def run_epoch(model, feature_pairs, optimizer, device, classification_weight, mmd_weight, swd_weight, moment_weight, training):
    model.tspn.train(training)
    model.classifier.train(training)
    totals = {"loss": 0.0, "classification": 0.0, "mmd": 0.0, "swd": 0.0, "moment": 0.0}
    correct = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for pair in tqdm(feature_pairs, desc="Training" if training else "Evaluating", leave=False):
            if training:
                optimizer.zero_grad(set_to_none=True)
            he = pair.he_features.to(device)
            real_ihc = pair.ihc_features.to(device)
            label = torch.tensor([pair.label], device=device, dtype=torch.long)
            synthetic_ihc = model.tspn(he)
            logits = model.classifier(he,synthetic_ihc,)
            classification = F.cross_entropy(logits.unsqueeze(0),label,)
            mmd = mmd_loss(synthetic_ihc,real_ihc,)
            swd = sliced_wasserstein(synthetic_ihc,real_ihc,)
            moment = moment_matching_loss(synthetic_ihc,real_ihc,)

            loss = (
                classification_weight * classification
                + mmd_weight * mmd
                + swd_weight * swd
                + moment_weight * moment
            )
            if training:
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.item())
            totals["classification"] += float(
                classification.item()
            )
            totals["mmd"] += float(mmd.item())
            totals["swd"] += float(swd.item())
            totals["moment"] += float(moment.item())
            correct += int(logits.argmax().item() == pair.label)
    count = max(len(feature_pairs), 1)
    return {key: value / count for key, value in totals.items()}, correct / count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config, train_pairs = build_feature_pairs_for_split(args, args.train_split)
    _, eval_pairs = build_feature_pairs_for_split(args, args.eval_split)

    tspn_checkpoint = Path(args.tspn_checkpoint)
    config = load_tspn_config_from_checkpoint(tspn_checkpoint, config)
    model = build_complete_pipeline(config, device)
    load_state_dict(model.tspn, tspn_checkpoint, device)
    load_state_dict(model.classifier, Path(args.classifier_checkpoint), device)
    unfreeze_module(model.tspn)
    unfreeze_module(model.classifier)
    classifier_lr = args.learning_rate if args.learning_rate is not None else args.classifier_learning_rate
    optimizer = torch.optim.AdamW(
        [
            {"params": model.tspn.parameters(), "lr": args.tspn_learning_rate},
            {"params": model.classifier.parameters(), "lr": classifier_lr},
        ],
        weight_decay=args.weight_decay,
    )

    history = []
    best_eval_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, train_acc = run_epoch(
            model, train_pairs, optimizer, device, args.classification_weight, args.mmd_weight, True
        )
        eval_metrics, eval_acc = run_epoch(
            model, eval_pairs, optimizer, device, args.classification_weight, args.mmd_weight, False
        )
        record: Dict[str, float] = {
            "epoch": float(epoch),
            "train_acc": train_acc,
            "eval_acc": eval_acc,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"eval_{key}": value for key, value in eval_metrics.items()},
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} | "
            f"eval_loss={eval_metrics['loss']:.6f} | eval_acc={eval_acc:.4f} | "
            f"eval_mmd={eval_metrics['mmd']:.6f}"
        )

        # Save only the model with the lowest evaluation loss
        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            best_epoch = epoch

            checkpoint_path = output_dir / "finetuned_pipeline_checkpoint.pt"

            torch.save(
                {
                    "tspn_state_dict": model.tspn.state_dict(),
                    "classifier_state_dict": model.classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "history": history,
                    "best_eval_loss": best_eval_loss,
                    "best_epoch": best_epoch,
                    "mmd_weight": args.mmd_weight,
                },
                checkpoint_path,
            )

            print(
                f"  ✓ New best model saved "
                f"(epoch={best_epoch}, eval_loss={best_eval_loss:.6f})"
            )

    
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved finetuned pipeline checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
