from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml
from pathlib import Path

@dataclass
class QualityConfig:
    green_contrast_threshold: float = 40.0
    failure_baseline: int = 3
    enable_denoise: bool = False

@dataclass
class IOConfig:
    input_images_dir: str
    preprocessed_images_dir: str
    labeled_map_csv: Optional[str] = None
    organized_output_dir: Optional[str] = None
    split_output_dir: Optional[str] = None
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])

@dataclass
class DatasetConfig:
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 32
    seed: int = 0
    interpolation: str = "bilinear"
    label_mode: str = "int"
    labels: str = "inferred"
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = True

@dataclass
class SplitConfig:
    enabled: bool = False
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    seed: int = 42

@dataclass
class VizConfig:
    enabled: bool = True
    out_dir: str = "outputs/visuals"

@dataclass
class SaveConfig:
    train_out: str = "outputs/processed/train"
    val_out: str = "outputs/processed/val"
    test_out: str = "outputs/processed/test"

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    level: str = "INFO"

@dataclass
class Config:
    io: IOConfig
    quality: QualityConfig = QualityConfig()
    dataset: DatasetConfig = DatasetConfig()
    split: SplitConfig = SplitConfig()
    viz: VizConfig = VizConfig()
    save: SaveConfig = SaveConfig()
    logging: LoggingConfig = LoggingConfig()

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    # Recursively map dicts to dataclasses
    def to_dc(dc_cls, d):
        # handle None case
        d = d or {}
        fields = {}
        for k, v in d.items():
            fields[k] = v
        return dc_cls(**fields)
    return Config(
        io=to_dc(IOConfig, raw.get("io")),
        quality=to_dc(QualityConfig, raw.get("quality")),
        dataset=to_dc(DatasetConfig, raw.get("dataset")),
        split=to_dc(SplitConfig, raw.get("split")),
        viz=to_dc(VizConfig, raw.get("viz")),
        save=to_dc(SaveConfig, raw.get("save")),
        logging=to_dc(LoggingConfig, raw.get("logging")),
    )
