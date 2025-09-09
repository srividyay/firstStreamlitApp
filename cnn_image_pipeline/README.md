# CNN Image Pipeline (Config-Driven, GitHub-Ready)

A minimal, **config-driven** TensorFlow image-classification training pipeline that can run locally (laptop/Colab) or in a production-like environment (server/VM) using the **same codebase** and different YAML configs.

## Features
- Clean project structure with YAML-based configs (`configs/local.yaml`, `configs/prod.yaml`)
- Toggle Google Drive mount (for Colab) via config
- Reproducible seeds, structured logging, artifact outputs (model, metrics, confusion matrix)
- Prefetch/cache tf.data pipeline, simple baseline CNN you can extend
- One-command run: `python -m src.train --config configs/local.yaml`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Local (edit data paths in configs/local.yaml)
python -m src.train --config configs/local.yaml

# Production (edit data/output paths in configs/prod.yaml)
python -m src.train --config configs/prod.yaml
```

### Config Overrides
You can override a few settings from the CLI:
```bash
python -m src.train --config configs/local.yaml --epochs 2 --dry_run true
```

## Expected Data Layout
```
<dataset_root>/
  train/
    class_a/...
    class_b/...
    ...
  val/
    class_a/...
    class_b/...
    ...
  test/
    class_a/...
    class_b/...
    ...
```
If you don’t have a `val/` split, set `validation_split` in the config and the pipeline will create it from `train/`.

## Artifacts
- Model: `artifacts/simple_cnn_model.keras` (SavedModel format)
- Metrics JSON: `artifacts/simple_cnn_metrics.json`
- Confusion matrix: `artifacts/simple_cnn_confusion_matrix.png`
- Logs: `artifacts/train.log`

## CI (optional)
A lightweight GitHub Actions workflow is included to sanity-check the pipeline on push using a short `--dry_run` job.
