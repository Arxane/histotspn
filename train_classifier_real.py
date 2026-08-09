from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from conch_runner import load_conch_model
from main import (
    PipelineConfig,
    build_conditioned_classifier,
    build_feature_pairs,
    build_patch_extractor,
    load_paired_wsi_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0: train the classifier on real HE + real IHC features."
    )
    parser.add_argument("--data-root", default="BCI_dataset")
    parser.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", default="artifacts/classifier_real")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
    parser.add_argument("--conch-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-start", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_pairs(args: argparse.Namespace, split: str, device: torch.device):
    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
        classifier_layers=args.classifier_layers,
    )
    tqdm.write(f"[Data Prep | {split}] Loading CONCH feature extractor...")
    conch_model, preprocess = load_conch_model()
    conch_model = conch_model.to(device).eval()
    samples = load_paired_wsi_samples(args.data_root, split)
    pairs = build_feature_pairs(
        samples,
        build_patch_extractor(config),
        conch_model,
        preprocess,
        device,
        config.conch_batch_size,
    )
    return config, pairs


def run_epoch(model, pairs, optimizer, criterion, device, training: bool):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for pair in tqdm(pairs, desc="Training" if training else "Evaluating", leave=False):
            he = pair.he_features.to(device)
            ihc = pair.ihc_features.to(device)
            label = torch.tensor([pair.label], device=device, dtype=torch.long)
            logits = model(he, ihc)
            loss = criterion(logits.unsqueeze(0), label)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_correct += int(logits.argmax(dim=-1).item() == pair.label)
    count = max(len(pairs), 1)
    return total_loss / count, total_correct / count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, train_pairs = build_pairs(args, "train", device)
    _, eval_pairs = build_pairs(args, args.eval_split, device)
    model = build_conditioned_classifier(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    best_acc = float("-inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Stage 0 epochs", unit="epoch"):
        train_loss, train_acc = run_epoch(model, train_pairs, optimizer, criterion, device, True)
        eval_loss, eval_acc = run_epoch(model, eval_pairs, optimizer, criterion, device, False)
        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        })
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | train_acc={train_acc:.4f} | "
            f"{args.eval_split}_loss={eval_loss:.6f} | {args.eval_split}_acc={eval_acc:.4f}"
        )
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        elif epoch >= args.early_stopping_start:
            stale_epochs += 1
        if epoch >= args.early_stopping_start and stale_epochs >= args.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Stage 0 did not produce a classifier checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "classifier_checkpoint.pt"
    torch.save(
        {
            "stage": 0,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "history": history,
            "best_eval_acc": best_acc,
            "best_epoch": best_epoch,
            "input_modalities": "real_he_and_real_ihc",
        },
        checkpoint_path,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved Stage 0 classifier checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
