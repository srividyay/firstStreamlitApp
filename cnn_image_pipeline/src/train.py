import argparse, json, os, random
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from .config import load_config
from .model import build_simple_cnn
from .data import build_datasets
from .utils.logger import build_logger

import gc, tensorflow as tf

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def maybe_enable_mixed_precision(enabled: bool):
    if enabled:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def train_and_eval(cfg: dict, overrides: dict):
    seed = int(cfg.get("seed", 42))
    set_seeds(seed)
    maybe_enable_mixed_precision(cfg.get("runtime", {}).get("mixed_precision", False))

    # Artifacts
    artifacts_dir = Path(cfg["artifacts"]["dir"])
    ensure_dir(artifacts_dir)
    log_file = artifacts_dir / "simple_cnn_train.log"
    logger = build_logger(log_file)
    logger.info(f"Loaded config for env: {cfg.get('env_name','unknown')}")

    # Build datasets
    
    train_ds, val_ds, test_ds, class_names = build_datasets(cfg)
    """# Backward compatible unpacking:
    if len(out) == 4:
        train_ds, val_ds, test_ds, class_names = out
    else:
        # legacy: function returned 3 values
        train_ds, val_ds, test_ds = out
        class_names = getattr(train_ds, "class_names", None)  # likely None on PrefetchDataset
        if class_names is None:
            # Fallback: infer from directory names
            data_dir = Path(cfg["data"].get("data_dir", ""))
            train_dir = Path(cfg["data"].get("train_dir", data_dir))
            class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])"""

    logger.info(f"Classes: {class_names}")
    
    logger.info("Datasets prepared")
    if train_ds is None:
        raise RuntimeError("Training dataset not found. Check your paths.")

    num_classes = int(cfg["model"]["num_classes"])
    input_h, input_w = cfg["data"]["image_size"]
    dropout = float(cfg["model"].get("dropout", 0.0))

    # Model
    model = build_simple_cnn(input_shape=(input_h, input_w, 3), num_classes=num_classes, dropout=dropout)
    model.compile(
        optimizer=cfg["model"].get("optimizer", "adam"),
        loss=cfg["model"].get("loss", "sparse_categorical_crossentropy"),
        metrics=cfg["model"].get("metrics", ["accuracy"]),
    )
    logger.info(model.summary())

    # Callbacks
    callbacks = []
    es_cfg = cfg.get("training", {}).get("early_stopping", {})
    if es_cfg.get("enabled", True):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            patience=int(es_cfg.get("patience", 3)),
            mode=es_cfg.get("mode", "min"),
            restore_best_weights=True
        ))

    epochs = int(overrides.get("epochs") or cfg["training"]["epochs"])
    dry_run = bool(overrides.get("dry_run")) or bool(cfg.get("dry_run", False))
    if dry_run:
        epochs = 1
        logger.info("Dry-run enabled: limiting epochs to 1")

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Metrics dict
    metrics = {
        "final_train_accuracy": float(history.history.get("accuracy", [None])[-1] or 0.0),
        "final_train_loss": float(history.history.get("loss", [None])[-1] or 0.0),
        "final_val_accuracy": float(history.history.get("val_accuracy", [None])[-1] if "val_accuracy" in history.history else None),
        "final_val_loss": float(history.history.get("val_loss", [None])[-1] if "val_loss" in history.history else None),
        "class_names": class_names,
    }

    # Evaluate on test set if available
    if test_ds is not None:
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        metrics["test_accuracy"] = float(test_acc)
        metrics["test_loss"] = float(test_loss)

        # Build confusion matrix
        y_true, y_pred = [], []
        for batch_x, batch_y in test_ds:
            preds = model.predict(batch_x, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1).tolist())
            y_true.extend(batch_y.numpy().tolist())

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        fig = plt.figure(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names if class_names else list(range(num_classes)))
        disp.plot(include_values=True, xticks_rotation=45)
        plt.tight_layout()
        cm_path = Path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["confusion_matrix_filename"]
        fig.savefig(cm_path)
        plt.close(fig)
        logger.info(f"Saved confusion matrix to {cm_path}")

        # Classification report (string)
        metrics["classification_report"] = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Save model
    model_path = Path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["save_model_filename"]
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Save metrics
    metrics_path = Path(cfg["artifacts"]["dir"]) / cfg["artifacts"]["save_metrics_filename"]
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    tf.keras.backend.clear_session()
    gc.collect()
    return metrics

def parse_args():
    ap = argparse.ArgumentParser(description="Train CNN with YAML config")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--epochs", type=int, help="Override epochs")
    ap.add_argument("--dry_run", type=str, help="Override dry_run (true/false)")
    return ap.parse_args()

def str_to_bool(x):
    if x is None:
        return None
    return str(x).lower() in {"1","true","yes","y","t"}

def main():
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {
        "epochs": args.epochs,
        "dry_run": str_to_bool(args.dry_run) if args.dry_run is not None else None,
    }
    train_and_eval(cfg, overrides)

if __name__ == "__main__":
    main()
