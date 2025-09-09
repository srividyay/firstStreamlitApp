# app.py
import os
import io
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

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
