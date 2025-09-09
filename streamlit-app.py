# app.py
import os
import io
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import sys, json

# ---------------- Settings ----------------
SAVE_DIR = st.secrets.get("SAVE_DIR", "uploads")
BRANCH = st.secrets.get("BRANCH", "main")

# Optional Git push config (set in .streamlit/secrets.toml)
GIT_REMOTE = st.secrets.get("GIT_REMOTE")  # e.g., "https://github.com/you/your-repo.git"
GIT_PAT = st.secrets.get("GIT_PAT")
GIT_USER_NAME = st.secrets.get("GIT_USER_NAME", "Uploader Bot")
GIT_USER_EMAIL = st.secrets.get("GIT_USER_EMAIL", "uploader@example.com")

ALLOWED_TYPES = ["csv", "xlsx", "xls"]
MAX_PREVIEW_ROWS = 100

"""PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_PACKAGE = st.secrets.get("TRAIN_PACKAGE", "cnn_image_pipeline.src")  # set to your package name"""

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_SRC_DIR = PROJECT_ROOT / "cnn_image_pipeline" / "src"
CONFIG_SRC_DIR = PROJECT_ROOT / "cnn_image_pipeline" / "configs"

# Make sure Python can import from src
if str(TRAINING_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_DIR))
if str(CONFIG_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_SRC_DIR))

# Point TRAIN_PACKAGE to match your repo structure
TRAIN_PACKAGE = "cnn_image_pipeline.src"
#CONFIG_PACKAGE = "cnn_image_pipeline.configs"

# Where to look for repo configs (customize this list)
CONFIG_DIRS = [
    PROJECT_ROOT / "cnn_image_processing" / "configs"
]
CONFIG_EXTS = (".yml", ".yaml")

def find_repo_configs():
    paths = []
    for d in CONFIG_DIRS:
        if d.exists():
            paths += sorted([p for p in d.rglob("*") if p.suffix.lower() in CONFIG_EXTS])
    return paths

def load_yaml_bytes(b: bytes):
    return yaml.safe_load(io.BytesIO(b).read())

def load_yaml_file(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

# ---------------- Helpers -----------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_stem(name: str) -> str:
    stem = os.path.splitext(name)[0]
    stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")).strip()
    return stem or "uploaded_file"

def short_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]

def build_filename(original_name: str, file_bytes: bytes) -> str:
    stem = safe_stem(original_name)
    ext = os.path.splitext(original_name)[1].lower()
    #ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    #h = short_hash(file_bytes)
    #return f"{stem}__{ts}__{h}{ext}"
    return f"{stem}"

def load_preview(file_bytes: bytes, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type")

def run_git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return e

def repo_is_git(cwd) -> bool:
    res = run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return not isinstance(res, subprocess.CalledProcessError) and res.stdout.strip() == "true"

def ensure_git_identity(cwd):
    subprocess.run(["git", "config", "user.name", GIT_USER_NAME], cwd=cwd)
    subprocess.run(["git", "config", "user.email", GIT_USER_EMAIL], cwd=cwd)

def ensure_remote(cwd):
    """Ensure a remote named origin exists and points to token-injected URL (when PAT present)."""
    if not GIT_REMOTE:
        return None, "No GIT_REMOTE configured."
    if not GIT_PAT:
        return None, "No GIT_PAT configured."

    token_remote = GIT_REMOTE
    if token_remote.startswith("https://"):
        token_remote = token_remote.replace("https://", f"https://{GIT_PAT}:x-oauth-basic@")

    rem_list = run_git(["remote"], cwd)
    if isinstance(rem_list, subprocess.CalledProcessError):
        return None, f"git remote failed: {rem_list.stderr}"

    remotes = rem_list.stdout.split()
    if "origin" in remotes:
        set_url = run_git(["remote", "set-url", "origin", token_remote], cwd)
        if isinstance(set_url, subprocess.CalledProcessError):
            return None, f"git remote set-url failed: {set_url.stderr}"
    else:
        add = run_git(["remote", "add", "origin", token_remote], cwd)
        if isinstance(add, subprocess.CalledProcessError):
            return None, f"git remote add failed: {add.stderr}"

    return "origin", None

def git_commit_file(repo_root: Path, path_to_add: Path):
    ensure_git_identity(repo_root)
    add_res = run_git(["add", str(path_to_add)], repo_root)
    if isinstance(add_res, subprocess.CalledProcessError):
        return False, f"git add failed: {add_res.stderr}"

    commit_msg = f"Add uploaded spreadsheet: {path_to_add.name}"
    commit_res = run_git(["commit", "-m", commit_msg], repo_root)
    if isinstance(commit_res, subprocess.CalledProcessError):
        return False, commit_res.stderr or "Nothing to commit."
    return True, "Committed."

def git_commit_all_under_save_dir(repo_root: Path):
    """Stage everything under SAVE_DIR and commit with a generic message."""
    ensure_git_identity(repo_root)
    add_res = run_git(["add", SAVE_DIR], repo_root)
    if isinstance(add_res, subprocess.CalledProcessError):
        return False, f"git add failed: {add_res.stderr}"
    commit_res = run_git(["commit", "-m", f"Commit pending uploads under {SAVE_DIR}/"], repo_root)
    if isinstance(commit_res, subprocess.CalledProcessError):
        return False, commit_res.stderr or "Nothing to commit."
    return True, "Committed pending uploads."

def git_push(repo_root: Path):
    remote_name, err = ensure_remote(repo_root)
    if err:
        return False, err
    run_git(["checkout", "-B", BRANCH], repo_root)  # create/switch to BRANCH
    push_res = run_git(["push", remote_name, f"HEAD:{BRANCH}"], repo_root)
    if isinstance(push_res, subprocess.CalledProcessError):
        return False, f"git push failed: {push_res.stderr}"
    return True, "Pushed to remote."

# ---------------- UI ---------------------
st.set_page_config(page_title="Spreadsheet Uploader (Git-enabled)", page_icon="📄", layout="centered")
st.title("📄 Spreadsheet Uploader (Git-enabled)")

repo_root = Path(__file__).resolve().parent

st.write(
    f"Upload a CSV or Excel file. It will be saved in `{SAVE_DIR}/` and you can optionally **commit** "
    f"and **push** the change to your Git repo."
)

with st.sidebar:
    st.header("Actions")
    do_commit = st.checkbox("Commit to Git after save", value=True)
    do_push = st.checkbox("Push to remote after commit", value=False,
                          help="Requires GIT_REMOTE and GIT_PAT in secrets.")
    st.caption(f"Branch: `{BRANCH}`")

    st.divider()
    st.subheader("Manual controls")
    if st.button("📝 Commit pending uploads"):
        if not repo_is_git(repo_root):
            st.warning("This directory is not a Git repository. (Run `git init` first.)")
        else:
            ok, msg = git_commit_all_under_save_dir(repo_root)
            if ok:
                st.success(msg)
            else:
                st.info(msg)

    if st.button("🚀 Push now"):
        if not repo_is_git(repo_root):
            st.warning("Not a Git repo; cannot push.")
        else:
            ok, msg = git_push(repo_root)
            if ok:
                st.success("Pushed to remote.")
            else:
                st.warning(f"Push note: {msg}")

uploaded = st.file_uploader("Choose a file", type=ALLOWED_TYPES, help="Supported: .csv, .xlsx, .xls")

if uploaded is not None:
    ext = os.path.splitext(uploaded.name)[1].lower()
    file_bytes = uploaded.getbuffer()

    with st.expander("Preview", expanded=True):
        try:
            df = load_preview(file_bytes, ext)
            if len(df) > MAX_PREVIEW_ROWS:
                st.info(f"Showing first {MAX_PREVIEW_ROWS} rows (of {len(df)} total).")
                st.dataframe(df.head(MAX_PREVIEW_ROWS))
            else:
                st.dataframe(df)
        except Exception as e:
            st.warning(f"Could not render preview: {e}")

    ensure_dir(SAVE_DIR)
    target_name = build_filename(uploaded.name, file_bytes)
    target_path = os.path.join(SAVE_DIR, target_name)

    try:
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        st.success(f"Saved to `{target_path}`")
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        st.stop()

    rel_path = Path(target_path).relative_to(repo_root)

    # --- Git Commit (per-upload) ---
    if do_commit:
        if not repo_is_git(repo_root):
            st.warning("This directory is not a Git repository. Skipping commit. (Run `git init` first.)")
        else:
            ok, msg = git_commit_file(repo_root, rel_path)
            if ok:
                st.success("Committed to Git.")
            else:
                st.info(f"Commit note: {msg}")

    # --- Git Push (per-upload) ---
    if do_commit and do_push:
        if not repo_is_git(repo_root):
            st.warning("Not a Git repo; cannot push.")
        else:
            ok, msg = git_push(repo_root)
            if ok:
                st.success("Pushed commit to remote.")
            else:
                st.warning(f"Push note: {msg}")

st.caption("Note: On Streamlit Cloud, local filesystem is ephemeral. Use the push option to persist uploads in your remote repo.")

# ------------------------- TRAINING PANEL --------------------------
with st.expander("🧠 Train a model", expanded=True):
    st.write("Use a YAML config from the repo or upload a new one.")

    # --- Source selector ---
    source_mode = st.radio(
        "Choose config source",
        ["Pick from repo", "Upload file"],
        horizontal=True,
    )

    # Repo picker
    selected_cfg_path = None
    if source_mode == "Pick from repo":
        repo_cfgs = find_repo_configs()
        if not repo_cfgs:
            st.info("No YAML configs found in configured folders. Try uploading instead.")
        else:
            labels = [str(p.relative_to(PROJECT_ROOT)) for p in repo_cfgs]
            choice = st.selectbox("Select a config from the repo", labels, index=0)
            selected_cfg_path = PROJECT_ROOT / choice

    # Upload
    uploaded_cfg = None
    if source_mode == "Upload file":
        uploaded_cfg = st.file_uploader("Upload YAML", type=["yml", "yaml"])

    # Common options
    epochs_override = st.number_input("Override epochs (optional)", min_value=1, step=1, value=10)
    dry_run = st.checkbox("Dry run (1 epoch)", value=False)
    artifacts_override = st.text_input(
        "Artifacts directory override (optional)",
        value="",
        help="If set, overrides cfg['artifacts']['dir'] before training.",
    )

    # Preview config
    st.markdown("**Config preview**")
    cfg_dict = None
    try:
        if source_mode == "Pick from repo" and selected_cfg_path:
            cfg_dict = load_yaml_file(selected_cfg_path)
        elif source_mode == "Upload file" and uploaded_cfg is not None:
            cfg_dict = load_yaml_bytes(uploaded_cfg.read())
        if cfg_dict is not None:
            st.json(cfg_dict)
    except Exception as e:
        st.warning(f"Could not parse YAML: {e}")

    # Train button
    if st.button("🚀 Train"):
        if source_mode == "Pick from repo" and not selected_cfg_path:
            st.error("Please select a config from the repo.")
            st.stop()
        if source_mode == "Upload file" and uploaded_cfg is None:
            st.error("Please upload a YAML config.")
            st.stop()

        # Persist uploaded config to disk so your loader can read a path
        if source_mode == "Upload file":
            CONFIGS_DIR = PROJECT_ROOT / "configs" / "uploads"
            CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            cfg_path = CONFIGS_DIR / uploaded_cfg.name
            cfg_bytes = uploaded_cfg.getvalue()
            cfg_path.write_bytes(cfg_bytes)
        else:
            cfg_path = selected_cfg_path

        with st.status("Starting training…", expanded=True) as status:
            try:
                # import training modules
                config_mod = __import__(f"{TRAIN_PACKAGE}.config", fromlist=["load_config"])
                train_mod = __import__(f"{TRAIN_PACKAGE}.train", fromlist=["train_and_eval"])

                # load config dict via your loader (ensures same parsing rules)
                cfg = config_mod.load_config(str(cfg_path))

                # optional artifacts override
                if artifacts_override.strip():
                    cfg.setdefault("artifacts", {})
                    cfg["artifacts"]["dir"] = artifacts_override.strip()

                overrides = {
                    "epochs": int(epochs_override) if epochs_override else None,
                    "dry_run": bool(dry_run),
                }

                st.write(f"Using config: `{cfg_path}`")
                metrics = train_mod.train_and_eval(cfg, overrides)

                st.success("Training finished.")
                st.subheader("Metrics")
                st.json(metrics)

                # Surface artifacts
                artifacts_dir = Path(cfg["artifacts"]["dir"])
                cm_file = artifacts_dir / cfg["artifacts"].get("confusion_matrix_filename", "confusion_matrix.png")
                if cm_file.exists():
                    st.image(str(cm_file), caption="Confusion Matrix", use_container_width=True)

                log_file = artifacts_dir / "simple_cnn_train.log"
                if log_file.exists():
                    st.download_button("Download training log", data=log_file.read_bytes(), file_name=log_file.name)

                model_file = artifacts_dir / cfg["artifacts"].get("save_model_filename", "model.keras")
                if model_file.exists():
                    st.download_button("Download model", data=model_file.read_bytes(), file_name=model_file.name)

                metrics_file = artifacts_dir / cfg["artifacts"].get("save_metrics_filename", "metrics.json")
                if metrics_file.exists():
                    st.download_button("Download metrics.json", data=metrics_file.read_bytes(), file_name=metrics_file.name)

                status.update(label="Done ✔️", state="complete")

            except Exception as e:
                st.exception(e)
                status.update(label="Training failed ❌", state="error")
