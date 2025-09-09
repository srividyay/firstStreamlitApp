from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import os
import shutil
from collections import Counter

import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .logger import get_logger

class ImagePreprocessor:
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = get_logger(__name__, self.cfg.logging.log_dir, self.cfg.logging.level)
        self.images_analysis: List[Dict[str, Dict[str, float]]] = []
        self.discarded: List[str] = []

    # -------------- Quality Metrics --------------
    def _brightness_gray(self, img: np.ndarray) -> Optional[int]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg = float(np.mean(gray))
        self.logger.debug(f"brightness_gray={avg:.2f}")
        if np.isnan(avg):
            return None
        if avg < 80: return -1
        if avg > 180: return 1
        return 0

    def _brightness_hsv(self, img: np.ndarray) -> Optional[int]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        val = float(np.mean(hsv[:, :, 2]))
        self.logger.debug(f"brightness_hsv={val:.2f}")
        if np.isnan(val):
            return None
        if val < 80: return -1
        if val > 180: return 1
        return 0

    def _green_contrast(self, img: np.ndarray) -> Optional[int]:
        _, g, _ = cv2.split(img)
        std = float(np.std(g))
        self.logger.debug(f"green_contrast_std={std:.2f}")
        if np.isnan(std):
            return None
        return 1 if std >= self.cfg.quality.green_contrast_threshold else 0

    def _noise_ok(self, img: np.ndarray) -> Optional[int]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray.astype(np.float32) - blurred.astype(np.float32)
        std = float(np.std(residual))
        self.logger.debug(f"noise_residual_std={std:.2f}")
        if np.isnan(std):
            return None
        # std > 50 => too noisy
        return 0 if std > 50 else 1

    def _analyze(self, img: np.ndarray, fname: str) -> int:
        b_g = self._brightness_gray(img)
        b_h = self._brightness_hsv(img)
        g_c = self._green_contrast(img)
        n_l = self._noise_ok(img)
        self.images_analysis.append({
            fname: {
                "brightness_gray": b_g,
                "brightness_hsv": b_h,
                "green_contrast_ok": g_c,
                "noise_ok": n_l
            }
        })
        fails = int((g_c == 0) + (b_g != 0) + (b_h != 0) + (n_l == 0))
        self.logger.info(f"{fname}: quality_fails={fails}")
        return fails

    # -------------- Transforms --------------
    def _normalize(self, img: np.ndarray) -> tf.Tensor:
        return tf.cast(img, tf.float32) / 255.0

    def _crop_square_resize(self, timg: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
        h, w = tf.shape(timg)[0], tf.shape(timg)[1]
        side = tf.minimum(h, w)
        oh = (h - side) // 2
        ow = (w - side) // 2
        cropped = tf.image.crop_to_bounding_box(timg, oh, ow, side, side)
        return tf.image.resize(cropped, size=size)

    def _brighten(self, timg: tf.Tensor, delta=0.2) -> tf.Tensor:
        return tf.image.adjust_brightness(timg, delta)

    def _dim(self, timg: tf.Tensor, delta=-0.2) -> tf.Tensor:
        return tf.image.adjust_brightness(timg, delta)

    def _contrast(self, timg: tf.Tensor, factor=2.0) -> tf.Tensor:
        return tf.image.adjust_contrast(timg, factor)

    def _saturate(self, timg: tf.Tensor, factor=1.5) -> tf.Tensor:
        return tf.image.adjust_saturation(timg, factor)

    def _denoise_gray(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    # -------------- Core --------------
    def process_image(self, img_bgr: np.ndarray, fname: str) -> Optional[tf.Tensor]:
        fails = self._analyze(img_bgr, fname)
        if fails > self.cfg.quality.failure_baseline:
            self.discarded.append(fname)
            self.logger.info(f"{fname}: discarded due to quality.")
            return None

        # Start with normalized tensor in [0,1]
        timg = self._normalize(img_bgr)

        # Adjust contrast based on green_contrast_ok
        details = next((d[fname] for d in self.images_analysis if fname in d), None)
        if details:
            if details.get("green_contrast_ok") == 1:
                timg = self._contrast(timg, 2.0)
            if details.get("brightness_gray") == -1:
                timg = self._brighten(timg, 0.2)
            elif details.get("brightness_gray") == 1:
                timg = self._dim(timg, -0.2)
            if details.get("brightness_hsv") == -1:
                timg = self._saturate(timg, 2.0)
            elif details.get("brightness_hsv") == 1:
                timg = self._saturate(timg, 0.5)

        # Optional denoise
        if self.cfg.quality.enable_denoise:
            den = self._denoise_gray(img_bgr)
            timg = tf.image.grayscale_to_rgb(tf.convert_to_tensor(den, dtype=tf.uint8))
            timg = self._normalize(timg)

        # Crop & resize
        timg = self._crop_square_resize(timg, self.cfg.dataset.image_size)
        return timg

    def preprocess_folder(self) -> None:
        in_dir = Path(self.cfg.io.input_images_dir)
        out_dir = Path(self.cfg.io.preprocessed_images_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in in_dir.iterdir() if p.suffix.lower() in self.cfg.io.exts]
        self.logger.info(f"Found {len(files)} images to process.")
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                self.logger.warning(f"Unreadable image: {p.name}")
                continue
            timg = self.process_image(img, p.name)
            if timg is None:
                continue
            tf.keras.utils.save_img(str(out_dir / p.name), timg)
        self.logger.info(f"Preprocessed images saved to {out_dir}")
        if self.discarded:
            self.logger.info(f"Discarded {len(self.discarded)} images: {self.discarded[:10]}{'...' if len(self.discarded)>10 else ''}")

    def organize_by_class(self) -> None:
        if not self.cfg.io.labeled_map_csv or not self.cfg.io.organized_output_dir:
            self.logger.info("Skipping organize_by_class (config not provided).")
            return
        df = pd.read_csv(self.cfg.io.labeled_map_csv)
        img_col = "image" if "image" in df.columns else df.columns[0]
        class_col = "class" if "class" in df.columns else df.columns[-1]

        src_dir = Path(self.cfg.io.preprocessed_images_dir)
        dst_base = Path(self.cfg.io.organized_output_dir)
        dst_base.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            fname = str(row[img_col])
            # ensure extension presence
            if Path(fname).suffix == "":
                fname += ".png"
            label = str(row[class_col])
            src = src_dir / fname
            dst_dir = dst_base / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst_dir / fname)
            else:
                self.logger.warning(f"Missing preprocessed file: {fname}")

        self.logger.info(f"Organized images written under {dst_base}")

    def split_train_val_test(self) -> None:
        if not self.cfg.split.enabled or not self.cfg.io.organized_output_dir or not self.cfg.io.split_output_dir:
            self.logger.info("Skipping split (config not provided or not enabled).")
            return
        import splitfolders
        splitfolders.ratio(
            input=self.cfg.io.organized_output_dir,
            output=self.cfg.io.split_output_dir,
            seed=self.cfg.split.seed,
            ratio=self.cfg.split.ratios,
        )
        self.logger.info(f"Split done -> {self.cfg.io.split_output_dir}")

    # ---------- TF Datasets & Viz ----------
    def _tfds_from_dir(self, path: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            directory=path,
            labels=self.cfg.dataset.labels,
            label_mode=self.cfg.dataset.label_mode,
            image_size=self.cfg.dataset.image_size,
            batch_size=self.cfg.dataset.batch_size,
            interpolation=self.cfg.dataset.interpolation,
            seed=self.cfg.dataset.seed,
            shuffle=shuffle,
        )

    def _class_hist(self, dataset, title: str, out_dir: Path):
        counts = Counter()
        for _, labels in dataset:
            lbls = labels.numpy()
            counts.update(lbls.tolist() if isinstance(lbls, np.ndarray) else [int(lbls)])
        cats = list(counts.keys())
        vals = list(counts.values())
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8,6))
        plt.bar(cats, vals)
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        fig_path = out_dir / f"{title.replace(' ','_').lower()}_class_distribution.png"
        plt.savefig(fig_path)
        plt.close()

    def datasets_and_visuals(self) -> None:
        if not (self.cfg.io.train_dir and self.cfg.io.val_dir and self.cfg.io.test_dir):
            self.logger.info("Skipping datasets (train/val/test dirs not specified).")
            return

        train_ds = self._tfds_from_dir(self.cfg.io.train_dir, self.cfg.dataset.shuffle_train)
        val_ds = self._tfds_from_dir(self.cfg.io.val_dir, self.cfg.dataset.shuffle_val)
        test_ds = self._tfds_from_dir(self.cfg.io.test_dir, self.cfg.dataset.shuffle_test)

        self.logger.info(f"Train spec: {train_ds.element_spec}")
        self.logger.info(f"Val spec:   {val_ds.element_spec}")
        self.logger.info(f"Test spec:  {test_ds.element_spec}")

        if self.cfg.viz.enabled:
            vdir = Path(self.cfg.viz.out_dir)
            self._class_hist(train_ds, "Train Dataset", vdir)
            self._class_hist(val_ds, "Val Dataset", vdir)
            self._class_hist(test_ds, "Test Dataset", vdir)

        # Optionally save out the TFDS as images for downstream pipelines
        self._save_ds_images(train_ds, Path(self.cfg.save.train_out))
        self._save_ds_images(val_ds, Path(self.cfg.save.val_out))
        self._save_ds_images(test_ds, Path(self.cfg.save.test_out))

    def _save_ds_images(self, dataset, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (imgs, labels) in enumerate(dataset):
            imgs_np = imgs.numpy()
            labels_np = labels.numpy()
            for j in range(imgs_np.shape[0]):
                lbl = int(labels_np[j])
                cls_dir = out_dir / f"{lbl}"
                cls_dir.mkdir(parents=True, exist_ok=True)
                fname = f"img_{i}_{j}_label{lbl}.png"
                tf.keras.utils.save_img(str(cls_dir / fname), imgs_np[j])
