# Retinal Preprocess Pipeline

Config-driven, minimal image preprocessing + dataset prep pipeline that runs locally or in CI, and is deployable from GitHub.

## Features
- YAML config for all paths and parameters
- Deterministic logging to console and file
- Quality checks (brightness, HSV value, green-channel contrast, noise)
- Conditional preprocessing (contrast/brightness/saturation/crop/normalize)
- Optional class folder organization from CSV mapping
- Train/Val/Test split using `split-folders` (optional)
- TF datasets export + quick visualizations (optional)
- CLI: `python -m retinal_pipeline --config configs/local.yaml`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m retinal_pipeline --config configs/local.yaml
```

## Config
See `configs/local.yaml` for an example. Override any field via env var or CLI `--override` (YAML snippet).

## Notes
- No Colab/Drive dependencies. Pure local paths.
- Keep images reasonably sized to avoid OOM on TF ops.

---

## Running with different configs

**Local (developer machine)**
```bash
python -m retinal_pipeline --config configs/local.yaml
```

**Production (container/VM)**
```bash
python -m retinal_pipeline --config configs/production.yaml
```

### Overriding without editing files
You can override any field inline:
```bash
python -m retinal_pipeline --config configs/production.yaml   --override 'logging: {level: DEBUG}'
```

Or point directories to mounted volumes:
```bash
python -m retinal_pipeline --config configs/production.yaml   --override 'io: {input_images_dir: "/mnt/vol/raw", preprocessed_images_dir: "/mnt/vol/pre"}'
```
