# HistoTSPN Pipeline

HistoTSPN learns to generate synthetic IHC CONCH features from HE whole-slide images. The final classifier uses HE features together with the generated IHC features, so the trained model can run when only an HE slide is available.

HE and IHC whole-slide images are paired by slide ID and label, but are not registered. Patch coordinates therefore have no correspondence across modalities. The pipeline never applies patch-level reconstruction, matching, or coordinate-based supervision.

## Architecture

All model inputs are CONCH patch features with shape `(number_of_patches, 512)`.

1. **TSPN:** A Transformer encoder contextualizes the variable-length HE patch set. A learnable matrix of `n_seeds` query slots, defaulting to `64`, attends to the contextualized HE features through multi-head cross-attention. A GELU MLP decoder produces a fixed synthetic IHC set with shape `(n_seeds, 512)`, independent of the number of HE patches.
2. **Classifier:** `DualBagClassifier` accepts independent variable-length HE and IHC patch sets. Separate ABMIL modules pool each modality internally, then a pooled feature interaction layer produces the slide-level logits.
3. **Inference compatibility:** `forward_pooled(he_vec, ihc_vec)` remains available for callers that already have one pooled vector per modality.

The default single-slide flow is:

```text
HE patches (N, 512)
  -> TSPN seed generation
synthetic IHC patches (64, 512)
  -> classifier ABMIL pooling and interaction
slide logits (4,)
```

## Current Training Protocol

The active pipeline has three training stages and one final test step:

1. **Stage 0:** Train a classifier on real HE + real IHC features. This is the real-modality teacher and an upper-bound reference.
2. **Stage 1:** Freeze the Stage 0 classifier and train TSPN from HE to synthetic IHC using feature fidelity, logit distillation, and classification loss.
3. **Stage 2:** Jointly fine-tune TSPN and the classifier with a small learning rate.
4. **Final test:** Evaluate once on the untouched test split.

The Stage 1 distribution objective is:

$$L = \alpha L_{MMD} + \beta L_{SWD} + \gamma L_{moment} + \delta L_{KL} + \eta L_{CE}.$$

`L_MMD`, `L_SWD`, and `L_moment` compare feature distributions without assuming patch correspondence. `L_KL` distills slide-level logits from the frozen real-IHC teacher, and `L_CE` supplies the hard-label task signal. The default weights are `alpha=1.0`, `beta=0.5`, `gamma=0.3`, `delta=2.0`, `eta=0.5`, and temperature `4.0`.

## Dataset Splits

[get_data.py](get_data.py) creates a deterministic virtual validation split:

- `train`: 90% of paired IDs from `BCI_dataset/*/train`.
- `validation`: 10% of paired train IDs, selected stratified by label with seed `42`.
- `test`: the original `BCI_dataset/*/test` files, unchanged.

No image files are copied. HE and IHC use the same held-out image IDs, and validation IDs are excluded from training. Training scripts use `validation` for checkpoint selection by default. [test_model.py](test_model.py) uses only `test`.

Expected dataset layout:

```text
BCI_dataset/
  HE/
    train/
    test/
  IHC/
    train/
    test/
```

HE and IHC filenames must share an image ID and end with a label in the range `0` to `3`.

## Setup

From the `histotspn` directory:

```bash
pip install -r requirements.txt
```

The first run may download CONCH weights through [conch_runner.py](conch_runner.py).

## Run The Pipeline

### Option A: Run all stages

```bash
cd /data/mayank_2023/histotspn

python main.py train-full \
  --data-root BCI_dataset \
  --output-base artifacts
```

This creates:

```text
artifacts/
  classifier_real/classifier_checkpoint.pt
  tspn_distill/tspn_checkpoint.pt
  final_stage/finetuned_pipeline_checkpoint.pt
```

### Option B: Run stages separately

#### Stage 0: Real HE + real IHC classifier

```bash
python train_classifier_real.py \
  --data-root BCI_dataset \
  --output-dir artifacts/classifier_real \
  --eval-split validation
```

The classifier checkpoint is trained with:

```text
classifier(real_he_features, real_ihc_features)
```

#### Stage 1: TSPN distillation

```bash
python train_tspn.py \
  --data-root BCI_dataset \
  --classifier-checkpoint artifacts/classifier_real/classifier_checkpoint.pt \
  --output-dir artifacts/tspn_distill \
  --eval-split validation \
  --alpha 1.0 \
  --beta 0.5 \
  --gamma 0.3 \
  --delta 2.0 \
  --eta 0.5 \
  --temperature 4.0 \
  --n-seeds 64 \
  --n-heads 8
```

The classifier is loaded before training, set to evaluation mode, and fully frozen. Each slide performs two classifier passes: real IHC features provide teacher logits, while TSPN-generated IHC features provide student logits. Only TSPN receives gradients.

#### Stage 2: Joint fine-tuning

```bash
python finetune_final.py \
  --data-root BCI_dataset \
  --tspn-checkpoint artifacts/tspn_distill/tspn_checkpoint.pt \
  --classifier-checkpoint artifacts/classifier_real/classifier_checkpoint.pt \
  --output-dir artifacts/final_stage \
  --train-split train \
  --eval-split validation \
  --tspn-learning-rate 1e-5 \
  --classifier-learning-rate 1e-4 \
  --mmd-weight 0.1
```

Stage 2 unfreezes both models and optimizes classification cross-entropy plus a small MMD regularizer between synthetic and real IHC feature distributions.

### Final test evaluation

Run this after selecting the final checkpoint. It reads the untouched test split and writes classification metrics plus optional pooled feature comparison statistics:

```bash
python test_model.py \
  --data-root BCI_dataset \
  --final-checkpoint artifacts/final_stage/finetuned_pipeline_checkpoint.pt \
  --output-dir test_outputs
```

Outputs:

```text
test_outputs/test_metrics.json
test_outputs/test_predictions.pt
```

If a final checkpoint is unavailable, `test_model.py` can use the Stage 1 TSPN checkpoint together with the Stage 0 classifier checkpoint:

```bash
python test_model.py \
  --tspn-checkpoint artifacts/tspn_distill/tspn_checkpoint.pt \
  --classifier-checkpoint artifacts/classifier_real/classifier_checkpoint.pt \
  --output-dir test_outputs
```

## Validation Analysis

To inspect whether synthetic IHC features retain useful information, run:

```bash
python verify_tspn.py \
  --tspn-checkpoint artifacts/tspn_distill/tspn_checkpoint.pt \
  --split validation \
  --output-dir validation_outputs
```

This produces linear-probe metrics, cosine-similarity metrics, feature arrays, and plots. Use `--split test` only for a final analysis after model selection.

## Sanity Check

Run the built-in variable-length shape check without loading CONCH or dataset files:

```bash
python main.py sanity-check
```

It tests HE length `37`, IHC length `51`, TSPN output `(64, 512)`, and classifier logits `(4,)`.

## Feature Export

Export synthetic IHC features:

```bash
python main.py export-tspn \
  --checkpoint artifacts/tspn_distill/tspn_checkpoint.pt \
  --split test \
  --output-dir artifacts/pipeline
```

The saved payload contains top-level keys:

```text
synthetic_ihc_patches: list of (n_seeds, 512) tensors
labels:                slide-level labels
sample_ids:            slide IDs
```

Real IHC patch features are retained without pooling in preparation artifacts and export payloads.

Export complete predictions:

```bash
python main.py export-complete \
  --tspn-checkpoint artifacts/tspn_distill/tspn_checkpoint.pt \
  --classifier-checkpoint artifacts/classifier_real/classifier_checkpoint.pt \
  --split test \
  --output-dir artifacts/pipeline
```

Use `--split validation` for inspection before final testing.

## File Guide

- [main.py](main.py): model builders, data preparation, export commands, and `train-full` orchestration.
- [get_data.py](get_data.py): paired HE/IHC loading and deterministic validation splitting.
- [get_patches.py](get_patches.py): tissue-aware patch extraction.
- [conch_runner.py](conch_runner.py): CONCH loading and patch encoding.
- [tspn.py](tspn.py): seed-based TSPN and distribution-matching losses.
- [conditioned_classifier.py](conditioned_classifier.py): ABMIL-pooled conditioned classifier.
- [train_classifier_real.py](train_classifier_real.py): Stage 0 teacher training.
- [train_tspn.py](train_tspn.py): Stage 1 frozen-classifier TSPN training.
- [train_tspn_distill.py](train_tspn_distill.py): compatibility entrypoint for Stage 1 training.
- [finetune_final.py](finetune_final.py): Stage 2 joint fine-tuning.
- [test_model.py](test_model.py): final test evaluation.
- [verify_tspn.py](verify_tspn.py): validation feature analysis.
- [distill_method/](distill_method/): archived legacy trainers; not used by the active pipeline.