from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    # Basic validation
    assert "data" in cfg and "root_dir" in cfg["data"], "Config missing data.root_dir"
    assert "model" in cfg and "num_classes" in cfg["model"], "Config missing model.num_classes"
    assert "artifacts" in cfg and "dir" in cfg["artifacts"], "Config missing artifacts.dir"
    return cfg
