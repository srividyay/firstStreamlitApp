from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def _maybe_mount_gdrive(enabled: bool, mount_path: str):
    if not enabled:
        return
    # Only attempt in Colab
    try:
        import google.colab  # type: ignore
        from google.colab import drive  # type: ignore
        drive.mount(mount_path, force_remount=True)
    except Exception:
        # Silent no-op outside Colab
        pass

def build_datasets(cfg: dict):
    _maybe_mount_gdrive(cfg.get("use_gdrive", False), cfg.get("gdrive_mount_path", "/content/drive"))

    data_cfg = cfg["data"]
    root_dir = Path(data_cfg["root_dir"])
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    batch_size = int(data_cfg.get("batch_size", 32))
    shuffle = bool(data_cfg.get("shuffle", True))
    interpolation = data_cfg.get("interpolation", "bilinear")
    validation_split = float(data_cfg.get("validation_split", 0.0))

    # Detect if explicit val/ exists
    has_explicit_val = (root_dir / "val").exists()
    train_dir = root_dir / "train"
    test_dir = root_dir / "test"
    val_dir = root_dir / "val" if has_explicit_val else None

    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train dir: {train_dir}")

    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        interpolation=interpolation,
        seed=cfg.get("seed", 42),
    )

    if has_explicit_val or validation_split <= 0.0:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=train_dir, shuffle=shuffle, **common
        )
        if val_dir and val_dir.exists():
            val_ds = tf.keras.utils.image_dataset_from_directory(
                directory=val_dir, shuffle=False, **common
            )
        else:
            val_ds = None
    else:
        # Create val from train via split
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=train_dir, shuffle=True, validation_split=validation_split,
            subset="training", **common
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=train_dir, shuffle=False, validation_split=validation_split,
            subset="validation", **common
        )

    test_ds = None
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=test_dir, shuffle=False, **common
        )
    """# Capture class_names BEFORE wrapping
    class_names = list(train_ds.class_names)"""
    
    # Performance: cache/prefetch
    num_calls = AUTOTUNE if str(cfg.get("runtime", {}).get("num_parallel_calls", "autotune")).lower() == "autotune" else int(cfg.get("runtime", {}).get("num_parallel_calls", 1))

    def _prep(ds):
        if ds is None:
            return None
        ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return _prep(train_ds), _prep(val_ds), _prep(test_ds)""", class_names"""
