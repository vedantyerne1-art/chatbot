import streamlit as st
import requests, json
import pandas as pd
from io import BytesIO, StringIO
from datetime import datetime
import fitz  # PyMuPDF

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Chat With File (Gemma2)", page_icon="🤖", layout="wide")
st.title("🤖 Chat With Your File (PDF / CSV) — Gemma2:2b (Ollama)")

MODEL = "gemma2:2b"
TEMPERATURE = 0.3
MAX_TOKENS = 1024
MEMORY_K = 8

FILE_CONTEXT_CHAR_LIMIT = 16000
CSV_PREVIEW_ROWS = 30


# ----------------------------
# OLLAMA STREAMING
# ----------------------------
def ollama_stream(prompt: str):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    }

    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                yield data["response"]
            if data.get("done"):
                break


# ----------------------------
# FILE READERS
# ----------------------------
def read_pdf_pymupdf(uploaded_file) -> str:
    """
    Reads PDF text using PyMuPDF.
    Works well for text PDFs.
    If PDF is fully scanned image, text may be empty.
    """
    try:
        pdf_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            t = page.get_text("text") or ""
            if t.strip():
                parts.append(t)
        doc.close()
        text = "\n\n".join(parts).strip()
        return text if text else "⚠️ No selectable text found in this PDF (it may be scanned/image-only)."
    except Exception as e:
        return f"⚠️ PDF read error: {e}"


def read_csv(uploaded_file):
    """
    Reads CSV into pandas DataFrame and returns (df, context_text).
    """
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")

    df = pd.read_csv(StringIO(text))

    # Build a compact context for the LLM
    info = []
    info.append("CSV SUMMARY")
    info.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    info.append(f"- Columns: {', '.join(map(str, df.columns.tolist()))}")

    dtypes = df.dtypes.astype(str).to_dict()
    dtype_lines = [f"  - {k}: {v}" for k, v in list(dtypes.items())[:60]]
    info.append("- Dtypes:\n" + "\n".join(dtype_lines))

    miss = df.isna().sum()
    miss_nonzero = miss[miss > 0]
    if len(miss_nonzero) > 0:
        miss_lines = [f"  - {idx}: {int(val)}" for idx, val in miss_nonzero.head(40).items()]
        info.append("- Missing values (non-zero):\n" + "\n".join(miss_lines))
    else:
        info.append("- Missing values: none")

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 0:
        desc = numeric.describe().transpose()
        desc = desc[["count", "mean", "std", "min", "max"]].head(25)
        info.append("- Numeric stats (top 25 numeric cols):\n" + desc.to_string())

    sample = df.head(CSV_PREVIEW_ROWS)
    info.append(f"- Sample first {min(CSV_PREVIEW_ROWS, len(df))} rows:\n{sample.to_string(index=False)}")

    return df, "\n".join(info)


# ----------------------------
# SESSION STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "file_type" not in st.session_state:
    st.session_state.file_type = ""

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

if "csv_df" not in st.session_state:
    st.session_state.csv_df = None


# ----------------------------
# TOP UI: UPLOAD + BUTTONS
# ----------------------------
uploaded = st.file_uploader("📎 Upload a PDF or CSV", type=["pdf", "csv"])

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat cleared ✅")

with col2:
    if st.session_state.messages:
        export_txt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            "⬇️ Export Chat (.txt)",
            data=export_txt,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

st.divider()


# ----------------------------
# LOAD FILE + BUILD CONTEXT
# ----------------------------
if uploaded is not None and uploaded.name != st.session_state.file_name:
    st.session_state.file_name = uploaded.name
    st.session_state.csv_df = None
    st.session_state.file_context = ""

    if uploaded.type == "application/pdf":
        st.session_state.file_type = "pdf"
        st.session_state.file_context = read_pdf_pymupdf(uploaded)
    else:
        st.session_state.file_type = "csv"
        try:
            df, context = read_csv(uploaded)
            st.session_state.csv_df = df
            st.session_state.file_context = context
        except Exception as e:
            st.session_state.file_context = f"⚠️ Could not read CSV. Error: {e}"

# Preview file
if st.session_state.file_name:
    st.success(f"Loaded file: {st.session_state.file_name} ✅")

    if st.session_state.file_type == "csv" and st.session_state.csv_df is not None:
        st.subheader("📊 CSV Preview")
        st.dataframe(st.session_state.csv_df.head(CSV_PREVIEW_ROWS), use_container_width=True)

    if st.session_state.file_type == "pdf":
        st.subheader("📄 Extracted PDF Text Preview")
        preview = (st.session_state.file_context or "")[:1200].strip()
        st.code(preview if preview else "(No text extracted.)")

st.divider()


# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# PROMPT BUILDER
# ----------------------------
def build_prompt(user_question: str) -> str:
    history = st.session_state.messages[-MEMORY_K:]
    hist_txt = ""
    for m in history:
        who = "User" if m["role"] == "user" else "Assistant"
        hist_txt += f"{who}: {m['content']}\n"

    file_context = st.session_state.file_context or ""
    if len(file_context) > FILE_CONTEXT_CHAR_LIMIT:
        file_context = file_context[-FILE_CONTEXT_CHAR_LIMIT:]

    file_note = ""
    if st.session_state.file_name:
        file_note = f"User uploaded file: {st.session_state.file_name} (type: {st.session_state.file_type})."

    return f"""
SYSTEM:
You are a helpful assistant.
If FILE_CONTEXT is provided, answer based ONLY on it.
If the answer is not in FILE_CONTEXT, say: "I can't find that in the uploaded file."
Be clear and structured.

{file_note}

FILE_CONTEXT:
{file_context}

CHAT_HISTORY:
{hist_txt}

USER_QUESTION:
{user_question}

ASSISTANT:
""".strip()


# ----------------------------
# CHAT INPUT
# ----------------------------
user_input = st.chat_input("Ask a question about your uploaded file...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not st.session_state.file_name:
        reply = "Please upload a PDF or CSV first, then ask your question about it."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""

            try:
                prompt = build_prompt(user_input)
                for token in ollama_stream(prompt):
                    full += token
                    placeholder.markdown(full)

                if not full.strip():
                    full = "⚠️ Empty response. Try asking again."
                    placeholder.markdown(full)

            except requests.exceptions.ConnectionError:
                full = "⚠️ Ollama is not running. Start it with: `ollama run gemma2:2b`"
                placeholder.markdown(full)
            except Exception as e:
                full = f"⚠️ Error: {e}"
                placeholder.markdown(full)

        st.session_state.messages.append({"role": "assistant", "content": full})