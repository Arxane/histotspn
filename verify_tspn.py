"""
ihc_feature_validation.py
--------------------------
Validates whether TSPN-generated synthetic IHC features encode non-redundant,
IHC-meaningful representations compared to raw HE CONCH features.

Metrics produced
----------------
1. Linear Probe Comparison - HE features vs. Synthetic IHC features to refute redundancy.
2. t-SNE / UMAP projection - Visual cluster separation of synthetic IHC features by HER2 label.
3. Intra-class vs. Inter-class Cosine Similarity - Quantifies manifold tightness.
4. Feature Activation Heatmaps - PCA breakdown of feature spaces across HER2 grades.

Usage
-----
python ihc_feature_validation.py \
    --tspn-checkpoint artifacts/tspn_distill/tspn_checkpoint.pt \
    --data-dir artifacts/pipeline \
    --split validation \
    --output-dir validation_outputs \
    --device cuda
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

PALETTE = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]
LABEL_NAMES = ["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"]
BG = "#0F1117"
FG = "#E8EAF0"
GRID = "#2A2D3A"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "text.color": FG,
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "font.family": "monospace",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    }
)

from biomed_runner import load_biomedclip_model

from main import (
    PipelineConfig,
    build_feature_pairs,
    build_patch_extractor,
    build_tspn_model,
    load_paired_wsi_samples,
    load_state_dict,
    load_tspn_config_from_checkpoint,
)


class FeaturePairDataset(Dataset):
    """Presents all extracted HE/IHC feature pairs as a batchable dataset."""

    def __init__(self, feature_pairs):
        self.feature_pairs = feature_pairs

    def __len__(self):
        return len(self.feature_pairs)

    def __getitem__(self, idx):
        record = self.feature_pairs[idx]
        he = record.he_features
        ihc = record.ihc_features
        if isinstance(he, np.ndarray):
            he = torch.from_numpy(he)
        if isinstance(ihc, np.ndarray):
            ihc = torch.from_numpy(ihc)

        label = record.label
        if isinstance(label, torch.Tensor):
            label = label.item()

        return {"he": he.float(), "ihc": ihc.float(), "label": torch.tensor(label, dtype=torch.long)}


def extract_feature_representations(tspn_model, loader, device):
    """Generates synthetic IHC features from HE features using trained TSPN."""

    def pool_feature_bag(features):
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features.mean(dim=0)
        return features

    tspn_model.eval()
    all_he_feats = []
    all_real_ihc_feats = []
    all_synth_ihc_feats = []
    all_labels = []

    for batch in tqdm(loader, desc="Generating Synthetic IHC Features"):
        he_feats = batch["he"].to(device)
        real_ihc_feats = batch["ihc"].to(device)
        labels = batch["label"]

        with torch.no_grad():
            synth_ihc_feats = tspn_model(he_feats)

            # Pool patch bags: [N, D] -> [D], or [B, N, D] -> [B, D].
            he_pooled = pool_feature_bag(he_feats)
            synth_ihc_pooled = pool_feature_bag(synth_ihc_feats)
            real_ihc_pooled = pool_feature_bag(real_ihc_feats)

        all_he_feats.append(he_pooled.cpu().float().reshape(1, -1))
        all_real_ihc_feats.append(real_ihc_pooled.cpu().float().reshape(1, -1))
        all_synth_ihc_feats.append(synth_ihc_pooled.cpu().float().reshape(1, -1))
        all_labels.append(labels.cpu().reshape(-1))

    he_array = torch.cat(all_he_feats, dim=0).numpy()
    real_ihc_array = torch.cat(all_real_ihc_feats, dim=0).numpy()
    synth_ihc_array = torch.cat(all_synth_ihc_feats, dim=0).numpy()
    labels_array = torch.cat(all_labels, dim=0).numpy()

    return he_array, real_ihc_array, synth_ihc_array, labels_array


def linear_probe(feats, labels):
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Linear probing requires at least two label classes")
    cv_folds = min(5, int(class_counts.min()))
    if cv_folds < 2:
        raise ValueError("Each label class needs at least two samples for cross-validation")

    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    cv_scores = cross_val_score(clf, X, labels, cv=cv_folds, scoring="accuracy")
    clf.fit(X, labels)
    preds = clf.predict(X)
    return cv_scores, preds, clf


def cosine_similarity_stats(feats, labels):
    """Intra-class vs inter-class mean cosine similarity."""
    norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    intra, inter = [], []
    for i in range(len(feats)):
        for j in range(i + 1, min(i + 50, len(feats))):
            sim = float(norms[i] @ norms[j])
            if labels[i] == labels[j]:
                intra.append(sim)
            else:
                inter.append(sim)
    return (float(np.mean(intra)) if intra else float("nan"), float(np.mean(inter)) if inter else float("nan"))


def generation_similarity_stats(generated, real):
    generated_norm = generated / (np.linalg.norm(generated, axis=1, keepdims=True) + 1e-8)
    real_norm = real / (np.linalg.norm(real, axis=1, keepdims=True) + 1e-8)
    cosine = np.sum(generated_norm * real_norm, axis=1)
    mse = np.mean((generated - real) ** 2, axis=1)
    return float(cosine.mean()), float(cosine.std()), float(mse.mean())


def plot_tsne(feats, labels, out_path):
    print("Running t-SNE projection...")
    if len(feats) < 4:
        raise ValueError("t-SNE requires at least four feature samples")
    pca = PCA(n_components=min(50, feats.shape[1]), random_state=42)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(feats))
    perplexity = min(30, max(2, (len(feats) - 1) // 3))
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, verbose=0)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000, verbose=0)
    X_2d = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=PALETTE[cls % len(PALETTE)],
            label=LABEL_NAMES[cls] if cls < len(LABEL_NAMES) else f"Class {cls}",
            alpha=0.7,
            s=18,
            linewidths=0,
        )
    ax.set_title("t-SNE of Generated Synthetic IHC Features", pad=12)
    ax.legend(framealpha=0.15, edgecolor=GRID)
    ax.grid(True)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out_path}")


def plot_probe_comparison(he_cv, ihc_cv, out_path):
    """Plots direct comparison between HE Linear Probe and Synthetic IHC Linear Probe."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [he_cv.mean(), ihc_cv.mean()]
    stds = [he_cv.std(), ihc_cv.std()]
    bars = ax.bar(["Raw HE Features", "Synthetic IHC Features"], means, yerr=stds, capsize=5, color=[PALETTE[0], PALETTE[1]], width=0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("Linear Probe: HE vs. Synthetic IHC Utility")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out_path}")


def plot_cosine_similarity(intra, inter, out_path):
    if not np.isfinite(intra) or not np.isfinite(inter):
        print("  Skipping cosine similarity plot: one similarity group is empty")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Intra-class", "Inter-class"], [intra, inter], color=[PALETTE[1], PALETTE[3]], width=0.4)
    ax.set_ylim(min(intra, inter) - 0.1, max(intra, inter) + 0.1)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Synthetic IHC Feature Separation")
    for bar, val in zip(bars, [intra, inter]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out_path}")


def plot_generation_similarity(cosine_mean, mse_mean, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    axes[0].bar(["Generated vs real"], [cosine_mean], color=PALETTE[1], width=0.4)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("Cosine similarity")
    axes[0].grid(True, axis="y")
    axes[1].bar(["Generated vs real"], [mse_mean], color=PALETTE[2], width=0.4)
    axes[1].set_ylabel("Mean squared error")
    axes[1].grid(True, axis="y")
    fig.suptitle("Generated IHC vs Real IHC Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out_path}")


def plot_summary_dashboard(metrics, out_path):
    fig = plt.figure(figsize=(10, 5), facecolor=BG)
    fig.suptitle("IHC Feature Quality - Validation Summary", fontsize=14, color=FG, y=1.01)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    he_mean = metrics["he_cv"].mean()
    ihc_mean = metrics["ihc_cv"].mean()
    bars = ax0.bar(["HE Raw", "IHC Synth"], [he_mean, ihc_mean], color=[PALETTE[0], PALETTE[1]], width=0.4)
    ax0.set_ylim(0, 1)
    ax0.set_title("Linear Probe Accuracy\n(HE vs Synthetic IHC)")
    ax0.set_ylabel("Accuracy")
    for bar, v in zip(bars, [he_mean, ihc_mean]):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax0.grid(True, axis="y")

    ax1 = fig.add_subplot(gs[1])
    vals = [metrics["intra_cosine"], metrics["inter_cosine"]]
    bars = ax1.bar(["Intra", "Inter"], vals, color=[PALETTE[1], PALETTE[3]], width=0.4)
    ax1.set_title("Cosine Similarity\n(Intra vs Inter class)")
    ax1.set_ylabel("Mean Cosine Sim")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out_path}")


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = PipelineConfig(
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        conch_batch_size=args.conch_batch_size,
    )

    print(f"Loading paired HE/IHC test samples from {args.data_root}...")
    samples = load_paired_wsi_samples(config.data_root, args.split)
    feature_model, preprocess, _ = load_biomedclip_model()
    feature_model = feature_model.to(device)
    feature_model.eval()
    extractor = build_patch_extractor(config)
    feature_pairs = build_feature_pairs(samples, extractor, feature_model, preprocess, device, config.conch_batch_size)
    print(f"Extracted features for {len(feature_pairs)} paired samples")

    print(f"Loading TSPN Checkpoint from {args.tspn_checkpoint}...")
    config = load_tspn_config_from_checkpoint(Path(args.tspn_checkpoint), PipelineConfig())
    tspn_model = build_tspn_model(config, device)
    load_state_dict(tspn_model, Path(args.tspn_checkpoint), device)
    tspn_model.eval()

    dataset = FeaturePairDataset(feature_pairs)

    # WSI bags have variable patch counts, so default tensor collation cannot
    # stack them into a batch. Process each pair without collation instead.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: batch[0])

    print("\n-- Feature Extraction & Generation -----------------------------")
    he_feats, real_ihc_feats, synth_ihc_feats, labels = extract_feature_representations(tspn_model, loader, device)
    print(f"  HE features shape       : {he_feats.shape}")
    print(f"  Real IHC shape          : {real_ihc_feats.shape}")
    print(f"  Synthetic IHC shape    : {synth_ihc_feats.shape}")
    print(f"  Labels shape           : {labels.shape}")

    print("\n-- Linear Probe Evaluation ------------------------------------")
    he_cv, he_preds, _ = linear_probe(he_feats, labels)
    ihc_cv, ihc_preds, _ = linear_probe(synth_ihc_feats, labels)

    print(f"  Raw HE Linear Probe CV Acc        : {he_cv.mean():.4f} - {he_cv.std():.4f}")
    print(f"  Synthetic IHC Linear Probe CV Acc : {ihc_cv.mean():.4f} - {ihc_cv.std():.4f}")

    print("\nSynthetic IHC Classification Report:")
    unique_labels = np.unique(labels)
    target_names = [LABEL_NAMES[label] if label < len(LABEL_NAMES) else f"Class {label}" for label in unique_labels]
    print(classification_report(labels, ihc_preds, labels=unique_labels, target_names=target_names, zero_division=0))

    print("\n-- Cosine Similarity Analysis ----------------------------------")
    intra, inter = cosine_similarity_stats(synth_ihc_feats, labels)
    print(f"  Intra-class Cosine Sim : {intra:.4f}")
    print(f"  Inter-class Cosine Sim : {inter:.4f}")
    print(f"  Separation Gap         : {intra - inter:.4f}")

    generation_cosine, generation_cosine_std, generation_mse = generation_similarity_stats(synth_ihc_feats, real_ihc_feats)
    print(f"  Generated/real Cosine  : {generation_cosine:.4f} - {generation_cosine_std:.4f}")
    print(f"  Generated/real MSE     : {generation_mse:.6f}")

    print("\n-- Generating Visualizations -----------------------------------")
    plot_tsne(real_ihc_feats, labels, out / "tsne_real_ihc.png")
    plot_tsne(synth_ihc_feats, labels, out / "tsne_synthetic_ihc.png")
    plot_probe_comparison(he_cv, ihc_cv, out / "linear_probe_comparison.png")
    plot_cosine_similarity(intra, inter, out / "cosine_similarity.png")
    plot_generation_similarity(generation_cosine, generation_mse, out / "generation_similarity.png")

    metrics = {
        "he_cv": he_cv,
        "ihc_cv": ihc_cv,
        "intra_cosine": intra,
        "inter_cosine": inter,
        "generation_cosine": generation_cosine,
        "generation_cosine_std": generation_cosine_std,
        "generation_mse": generation_mse,
    }
    plot_summary_dashboard(metrics, out / "summary_dashboard.png")

    np.save(out / "he_features.npy", he_feats)
    np.save(out / "real_ihc_features.npy", real_ihc_feats)
    np.save(out / "synth_ihc_features.npy", synth_ihc_feats)
    np.save(out / "labels.npy", labels)
    print(f"\n Validation outputs saved to '{out}/'")

    probe_delta = ihc_cv.mean() - he_cv.mean()
    print("\n-- Validation Diagnosis -----------------------------------------")
    if probe_delta > 0.02:
        print(f"  POSITIVE RESULT: Synthetic IHC features improve linear separability over raw HE features (+{probe_delta:.3f} accuracy).")
        print("  This refutes the redundancy critique: TSPN successfully disentangled biomarker signals.")
    elif abs(probe_delta) <= 0.02:
        print("  NEUTRAL RESULT: Synthetic IHC features match raw HE features in linear separability.")
        print("  Recommendation: Ensure Stage 3 joint fine-tuning is executed to optimize cross-attention weights for classification.")
    else:
        print(f"  WARNING: Synthetic IHC features underperform raw HE features ({probe_delta:.3f}).")
        print("  Recommendation: Re-check Stage 1 distribution-loss weights in 'train_tspn.py'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TSPN synthetic IHC feature quality.")
    parser.add_argument("--tspn-checkpoint", default="artifacts/tspn_distill/tspn_checkpoint.pt", help="Path to trained TSPN checkpoint file (.pt)")
    parser.add_argument("--data-root", default="BCI_dataset")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.5)
    parser.add_argument("--conch-batch-size", type=int, default=32)
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate (validation by default; use test only for final analysis)",
    )
    parser.add_argument("--output-dir", default="validation_outputs", help="Directory to store validation plots and arrays")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
