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

def build_datasets(cfg):
    data_cfg = cfg["data"]
    data_dir = Path(data_cfg["root_dir"])
    img_h, img_w = data_cfg["image_size"]
    batch_size = int(data_cfg.get("batch_size", 32))
    seed = int(cfg.get("seed", 42))
    val_split = float(data_cfg.get("val_split", 0.2))
    shuffle = bool(data_cfg.get("shuffle", True))

    # NEW: memory knobs with safe defaults
    cache_mode = data_cfg.get("cache", "disk")   # "disk", "memory", or False
    prefetch_buf = int(data_cfg.get("prefetch_buffer", 1))  # 1 is memory-friendly
    shuffle_buf = int(data_cfg.get("shuffle_buffer", max(batch_size * 2, 64)))
    num_calls   = data_cfg.get("num_parallel_calls", None)  # None => let TF decide conservatively

    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_h, img_w),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_h, img_w),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_raw.class_names)

    # Optional mapping step? (keep parallelism modest)
    # if you have a map(), do: .map(fn, num_parallel_calls=num_calls)

    # Cache policy
    if cache_mode:
        if str(cache_mode).lower() == "memory":
            train_raw = train_raw.cache()
            val_raw   = val_raw.cache()
        else:
            # cache to disk under artifacts to avoid RAM spikes
            artifacts_dir = Path(cfg["artifacts"]["dir"])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            train_raw = train_raw.cache(str(artifacts_dir / "tfdata_train.cache"))
            val_raw   = val_raw.cache(str(artifacts_dir / "tfdata_val.cache"))

    # Shuffle (smaller buffer)
    if shuffle:
        train_raw = train_raw.shuffle(buffer_size=shuffle_buf, seed=seed)

    # Prefetch with small buffer to limit resident memory
    train_ds = train_raw.prefetch(prefetch_buf)
    val_ds   = val_raw.prefetch(prefetch_buf)

    test_ds = None
    test_dir = data_cfg.get("test_dir")
    if test_dir:
        test_raw = tf.keras.utils.image_dataset_from_directory(
            Path(test_dir),
            labels="inferred",
            label_mode="int",
            image_size=(img_h, img_w),
            batch_size=batch_size,
            shuffle=False,
        )
        test_ds = test_raw.prefetch(prefetch_buf)

    return train_ds, val_ds, test_ds, class_names
