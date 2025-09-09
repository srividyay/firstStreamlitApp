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
