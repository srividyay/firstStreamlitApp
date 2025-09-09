# nlp-app.py
# Streamlit app: Upload CSV → run data pipeline → download processed CSV
#               → run model pipeline → show sentiment score + paragraph

import io
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


# ============================
# ====== WIRING POINTS =======
# ============================
# 👉 If your pipelines are Python-callable, set IMPORT_CALLABLES = True and
#    update the import paths & function names below.
# 👉 If they’re CLI scripts, set IMPORT_CALLABLES = False and set the CLI paths.

IMPORT_CALLABLES = True  # set False if you only have CLI entrypoints

# --- Python callable mode (edit these to match your repo) ---
if IMPORT_CALLABLES:
    try:
        # Example: from myrepo.pipelines.data import run as data_run
        #          from myrepo.pipelines.model import predict as model_predict
        from data_pipeline.process import run as data_run          # <-- EDIT
        from model_pipeline.predict import run as model_predict    # <-- EDIT
        PIPELINES_OK = True
        PIPELINES_MODE = "python"
    except Exception as e:
        PIPELINES_OK = False
        PIPELINES_MODE = "python"
        _import_err = e

# --- CLI mode (edit these to match your environment) ---
DATA_PIPELINE_CLI = "python -m data_pipeline.process"   # e.g., "python scripts/data_pipeline.py"
MODEL_PIPELINE_CLI = "python -m model_pipeline.predict" # e.g., "python scripts/model_pipeline.py"


# ============================
# ===== Helper functions =====
# ============================

def _run_data_pipeline_python(input_csv_path: Path, **kwargs) -> pd.DataFrame:
    """
    Expects your callable to accept input path and return either:
      - a pandas.DataFrame, or
      - a path to a processed CSV
    """
    result = data_run(input_path=str(input_csv_path), **kwargs)
    if isinstance(result, pd.DataFrame):
        return result
    # Assume result is path-like
    out_path = Path(result)
    return pd.read_csv(out_path)


def _run_model_pipeline_python(df: pd.DataFrame, text_col: str, **kwargs) -> pd.DataFrame:
    """
    Expects your callable to accept a DataFrame (and text column) and return
    a DataFrame with at least: ['sentiment_score', 'sentiment_text'].
    """
    result = model_predict(df=df, text_col=text_col, **kwargs)
    if isinstance(result, pd.DataFrame):
        return result
    # If it returns a path:
    out_path = Path(result)
    return pd.read_csv(out_path)


def _run_cli(cmd: str, args: list) -> Tuple[int, str, str]:
    """Run a CLI command and return (returncode, stdout, stderr)."""
    full_cmd = cmd.split() + args
    proc = subprocess.run(full_cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _run_data_pipeline_cli(input_csv_path: Path, output_csv_path: Path, **kwargs) -> pd.DataFrame:
    """
    Expects your CLI to support: --input <path> --output <path>
    Add/alter flags as needed.
    """
    args = ["--input", str(input_csv_path), "--output", str(output_csv_path)]
    for k, v in kwargs.items():
        args += [f"--{k.replace('_','-')}", str(v)]
    code, out, err = _run_cli(DATA_PIPELINE_CLI, args)
    if code != 0:
        raise RuntimeError(f"Data pipeline failed.\nSTDOUT:\n{out}\n\nSTDERR:\n{err}")
    return pd.read_csv(output_csv_path)


def _run_model_pipeline_cli(input_csv_path: Path, output_csv_path: Path, text_col: str, **kwargs) -> pd.DataFrame:
    """
    Expects your CLI to support: --input <path> --output <path> --text-col <name>
    """
    args = ["--input", str(input_csv_path), "--output", str(output_csv_path), "--text-col", text_col]
    for k, v in kwargs.items():
        args += [f"--{k.replace('_','-')}", str(v)]
    code, out, err = _run_cli(MODEL_PIPELINE_CLI, args)
    if code != 0:
        raise RuntimeError(f"Model pipeline failed.\nSTDOUT:\n{out}\n\nSTDERR:\n{err}")
    return pd.read_csv(output_csv_path)


@st.cache_data(show_spinner=False)
def _read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _download_df_button(df: pd.DataFrame, label: str, filename: str, key: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key
    )


# ============================
# ========= UI/App ===========
# ============================

st.set_page_config(page_title="Sentiment Pipeline", page_icon="💬", layout="wide")
st.title("💬 Sentiment Processing & Inference")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        1) Upload a CSV containing at least one **text** column.  
        2) We run your **data-processing pipeline** to clean/normalize the data.  
        3) We run your **model pipeline** to produce:
           - a numeric **sentiment_score** (e.g., -1 to 1 or 0 to 1), and  
           - a **sentiment_text** paragraph (short explanation).  
        4) You can preview and download both processed and predicted outputs.
        """
    )

# Sidebar options
st.sidebar.header("Options")
text_col = st.sidebar.text_input("Text column name", value="text")
show_preview_rows = st.sidebar.number_input("Preview rows", min_value=5, max_value=200, value=25, step=5)
run_data_kwargs = st.sidebar.text_area(
    "Extra data-pipeline kwargs (key=value per line)",
    value=""
)
run_model_kwargs = st.sidebar.text_area(
    "Extra model-pipeline kwargs (key=value per line)",
    value=""
)

def _parse_kwargs(raw: str) -> dict:
    kv = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv

data_kwargs = _parse_kwargs(run_data_kwargs)
model_kwargs = _parse_kwargs(run_model_kwargs)

# File uploader
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to get started.")
    if PIPELINES_MODE == "python" and not PIPELINES_OK:
        st.warning(f"Could not import your pipeline callables. Edit the WIRING section.\n\nImport error: {_import_err}")
    st.stop()

# Read initial CSV
try:
    df_raw = _read_uploaded_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.subheader("📥 Uploaded Data (preview)")
st.dataframe(df_raw.head(show_preview_rows), use_container_width=True)

if text_col not in df_raw.columns:
    st.error(f"Text column '{text_col}' not found in uploaded CSV. Columns: {list(df_raw.columns)}")
    st.stop()

# Run pipelines
run_btn = st.button("▶️ Run Pipelines")
if not run_btn:
    st.stop()

with st.status("Running data-processing pipeline...", expanded=False):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.csv"
            df_raw.to_csv(input_path, index=False)

            if IMPORT_CALLABLES and PIPELINES_OK:
                df_processed = _run_data_pipeline_python(input_path, **data_kwargs)
            else:
                # Fallback to CLI mode
                df_processed = _run_data_pipeline_cli(
                    input_csv_path=input_path,
                    output_csv_path=tmpdir / "processed.csv",
                    **data_kwargs
                )
    except Exception as e:
        st.error(f"Data pipeline error: {e}")
        st.stop()

st.success("Data processing complete ✅")

st.subheader("🧹 Processed Data (preview)")
st.dataframe(df_processed.head(show_preview_rows), use_container_width=True)
_download_df_button(df_processed, "⬇️ Download processed CSV", "processed.csv", key="dl_processed")

# Ensure text column survives
if text_col not in df_processed.columns:
    st.warning(
        f"The processed data does not contain the text column '{text_col}'. "
        "Please verify your data pipeline preserves or renames it accordingly."
    )
    st.stop()

with st.status("Running model pipeline for sentiment...", expanded=False):
    try:
        if IMPORT_CALLABLES and PIPELINES_OK:
            df_pred = _run_model_pipeline_python(df_processed, text_col=text_col, **model_kwargs)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                proc_path = tmpdir / "proc.csv"
                out_path = tmpdir / "pred.csv"
                df_processed.to_csv(proc_path, index=False)
                df_pred = _run_model_pipeline_cli(
                    input_csv_path=proc_path,
                    output_csv_path=out_path,
                    text_col=text_col,
                    **model_kwargs
                )

        # Expect at least these columns; adapt if your model returns different names
        expected_cols = {"sentiment_score", "sentiment_text"}
        missing = expected_cols - set(df_pred.columns)
        if missing:
            # Try common alternatives to be helpful:
            alt_map = {
                "sentiment_score": ["score", "sentiment", "probability"],
                "sentiment_text": ["explanation", "summary", "rationale", "sentiment_paragraph"],
            }
            for need in list(missing):
                for alt in alt_map.get(need, []):
                    if alt in df_pred.columns:
                        df_pred[need] = df_pred[alt]
                        missing.remove(need)
                        break
        if missing:
            raise ValueError(f"Model output missing columns: {missing}. Got columns: {list(df_pred.columns)}")

        # Attach to processed frame if not already aligned (best-effort by index)
        if len(df_pred) == len(df_processed):
            out_df = df_processed.copy()
            out_df["sentiment_score"] = df_pred["sentiment_score"].values
            out_df["sentiment_text"] = df_pred["sentiment_text"].values
        else:
            # Fall back to showing the model output alone
            out_df = df_pred.copy()

    except Exception as e:
        st.error(f"Model pipeline error: {e}")
        st.stop()

st.success("Sentiment inference complete ✅")

# Summary metrics (if numeric)
col1, col2 = st.columns(2)
with col1:
    if pd.api.types.is_numeric_dtype(out_df["sentiment_score"]):
        st.metric("Average sentiment score", f"{out_df['sentiment_score'].mean():.4f}")
with col2:
    st.write(" ")

st.subheader("🧠 Predictions (preview)")
preview_cols = [c for c in out_df.columns if c not in ("sentiment_text",)]
st.dataframe(
    out_df[preview_cols].head(show_preview_rows),
    use_container_width=True
)
st.write("**Sentiment explanation (first few rows):**")
for i, row in out_df.head(min(5, len(out_df))).iterrows():
    st.markdown(f"**Row {i}** — score: `{row['sentiment_score']}`")
    st.write(row["sentiment_text"])
    st.markdown("---")

_download_df_button(out_df, "⬇️ Download predictions CSV", "predictions.csv", key="dl_predictions")

st.success("Done!")
