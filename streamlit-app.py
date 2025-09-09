# app.py
import os
import io
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

# ---- Settings ---------------------------------------------------------------
# You can override SAVE_DIR via .streamlit/secrets.toml with:
# SAVE_DIR = "some/other/folder"
SAVE_DIR = st.secrets.get("SAVE_DIR", "uploads")

ALLOWED_TYPES = ["csv", "xlsx", "xls"]  # what we’ll accept
MAX_PREVIEW_ROWS = 100  # limit preview size

# ---- Helpers ----------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_stem(name: str) -> str:
    # Make a filesystem-friendly stem
    stem = os.path.splitext(name)[0]
    stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")).strip()
    return stem or "uploaded_file"

def short_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]

def build_filename(original_name: str, file_bytes: bytes) -> str:
    stem = safe_stem(original_name)
    ext = os.path.splitext(original_name)[1].lower()  # includes leading dot
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

# ---- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Spreadsheet Uploader", page_icon="📄", layout="centered")
st.title("📄 Spreadsheet Uploader")
st.write(
    "Upload a CSV or Excel file. It will be saved inside your repo under "
    f"`{SAVE_DIR}/` with a unique, timestamped filename."
)

uploaded = st.file_uploader(
    "Choose a file",
    type=ALLOWED_TYPES,
    help="Supported: .csv, .xlsx, .xls",
)

if uploaded is not None:
    # Validate type
    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext.replace(".", "") not in ALLOWED_TYPES:
        st.error("Unsupported file type.")
        st.stop()

    # Read bytes once
    file_bytes = uploaded.getbuffer()

    # Preview (optional)
    with st.expander("Preview", expanded=True):
        try:
            df = load_preview(file_bytes, ext)
            # Limit preview for huge files
            if len(df) > MAX_PREVIEW_ROWS:
                st.info(f"Showing first {MAX_PREVIEW_ROWS} rows (of {len(df)} total).")
                st.dataframe(df.head(MAX_PREVIEW_ROWS))
            else:
                st.dataframe(df)
        except Exception as e:
            st.warning(f"Could not render preview: {e}")

    # Save
    ensure_dir(SAVE_DIR)
    target_name = build_filename(uploaded.name, file_bytes)
    target_path = os.path.join(SAVE_DIR, target_name)

    try:
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        st.success(f"Saved to `{target_path}`")
        st.caption("Tip: commit this folder to your repo or add it to .gitignore if you prefer not to track uploads.")
    except Exception as e:
        st.error(f"Failed to save file: {e}")

# ---- Footer -----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.write(f"**Save directory:** `{SAVE_DIR}`")
    st.write("To change this, set `SAVE_DIR` in `.streamlit/secrets.toml`.")
    st.write("Note: On Streamlit Cloud, filesystem writes are ephemeral—consider cloud storage for persistence.")