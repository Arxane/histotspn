from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from tqdm import tqdm

from main import (
    PipelineConfig,
    build_conditioned_classifier,
    build_feature_pairs,
    build_patch_extractor,
    build_tspn_model,
    freeze_module,
    load_paired_wsi_samples,
    load_state_dict,
)
from tspn import tspn_distribution_loss
from biomed_runner import load_biomedclip_model

import hashlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train seed-based TSPN with frozen classifier distillation.")
    parser.add_argument("--data-root", type=str, default="BCI_dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts/tspn_training")
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--tspn-checkpoint", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default="validation", choices=("validation", "test"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
    parser.add_argument("--conch-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-start", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--n-seeds", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_pairs(args, split, device):
    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
        n_seeds=args.n_seeds,
        output_size=args.n_seeds,
        n_heads=args.n_heads,
        nhead=args.n_heads,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    feature_model, preprocess, _ = load_biomedclip_model()
    feature_model = feature_model.to(device).eval()
    samples = load_paired_wsi_samples(args.data_root, split)
    cfg_hash   = hashlib.md5(
        f"{args.patch_size}_{args.stride}_{args.min_tissue_fraction}".encode()
    ).hexdigest()[:8]
    cache_root = Path("feature_cache") / cfg_hash / split
    pairs = build_feature_pairs(
        samples,
        build_patch_extractor(config),
        feature_model,
        preprocess,
        device,
        config.conch_batch_size,
        cache_root=cache_root,
    )
    return config, pairs


def run_epoch(model, classifier, pairs, optimizer, device, args, training):
    model.train(training)
    classifier.eval()
    totals = {key: 0.0 for key in ("loss", "mmd", "swd", "mom", "kl", "task")}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for pair in tqdm(pairs, desc="Training" if training else "Evaluating", leave=False):
            if training:
                optimizer.zero_grad(set_to_none=True)
            he = pair.he_features.to(device)
            real_ihc = pair.ihc_features.to(device)
            label = torch.tensor([pair.label], device=device, dtype=torch.long)
            syn_ihc = model(he)
            logits_real = classifier(he, real_ihc)
            logits_syn = classifier(he, syn_ihc)
            loss, components = tspn_distribution_loss(
                syn_ihc, real_ihc, logits_syn, logits_real, label,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                delta=args.delta, eta=args.eta, temperature=args.temperature,
            )
            if training:
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.item())
            for key, value in components.items():
                totals[key] += value
    count = max(len(pairs), 1)
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(model, optimizer, output_dir, epoch, best_val_loss, args, metrics):
    path = output_dir / "tspn_checkpoint.pt"
    torch.save({
        "tspn_state_dict": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "loss_weights": {
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "delta": args.delta, "eta": args.eta,
        },
        "n_seeds": args.n_seeds,
        "config": vars(args),
        "metrics": metrics,
    }, path)
    return path


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_checkpoint = Path(args.classifier_checkpoint)
    config, train_pairs = build_pairs(args, "train", device)
    _, val_pairs = build_pairs(args, args.eval_split, device)
    classifier = build_conditioned_classifier(config, device)
    load_state_dict(classifier, classifier_checkpoint, device)
    freeze_module(classifier)
    model = build_tspn_model(config, device)
    if args.tspn_checkpoint:
        load_state_dict(model, Path(args.tspn_checkpoint), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, classifier, train_pairs, optimizer, device, args, True)
        val_metrics = run_epoch(model, classifier, val_pairs, optimizer, device, args, False)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print("Epoch %03d | train loss %.5f | val loss %.5f | mmd %.5f | swd %.5f | mom %.5f | kl %.5f | task %.5f" % (
            epoch, train_metrics["loss"], val_metrics["loss"], val_metrics["mmd"], val_metrics["swd"], val_metrics["mom"], val_metrics["kl"], val_metrics["task"]))
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        elif epoch >= args.early_stopping_start:
            stale += 1
        if epoch >= args.early_stopping_start and stale >= args.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("No TSPN checkpoint was produced")
    model.load_state_dict(best_state)
    checkpoint = save_checkpoint(model, optimizer, output_dir, best_epoch, best_val_loss, args, history)
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
