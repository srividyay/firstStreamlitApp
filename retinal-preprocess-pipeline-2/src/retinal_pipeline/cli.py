import argparse
import sys
import yaml

from .config import load_config, Config
from .preprocess import ImagePreprocessor

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Retinal Preprocess Pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--override", help="Inline YAML to override config (e.g. 'logging: {level: DEBUG}')")
    return p.parse_args(argv)

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d

def main(argv=None):
    args = parse_args(argv)
    with open(args.config, "r") as f:
        base = yaml.safe_load(f) or {}
    if args.override:
        override = yaml.safe_load(args.override) or {}
        base = deep_update(base, override)

    cfg = load_config_from_raw(base)
    runner = ImagePreprocessor(cfg)

    runner.preprocess_folder()
    runner.organize_by_class()
    runner.split_train_val_test()
    runner.datasets_and_visuals()
    return 0

def load_config_from_raw(raw: dict) -> Config:
    # Reuse loader but from in-memory dict
    from .config import Config, IOConfig, QualityConfig, DatasetConfig, SplitConfig, VizConfig, SaveConfig, LoggingConfig
    def to_dc(dc_cls, d):
        d = d or {}
        return dc_cls(**d)
    return Config(
        io=to_dc(IOConfig, raw.get("io")),
        quality=to_dc(QualityConfig, raw.get("quality")),
        dataset=to_dc(DatasetConfig, raw.get("dataset")),
        split=to_dc(SplitConfig, raw.get("split")),
        viz=to_dc(VizConfig, raw.get("viz")),
        save=to_dc(SaveConfig, raw.get("save")),
        logging=to_dc(LoggingConfig, raw.get("logging")),
    )

if __name__ == "__main__":
    raise SystemExit(main())
