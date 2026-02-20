import streamlit as st
import requests
import json
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import pandas as pd

# ==========================
# CONFIG
# ==========================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma2:2b"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Smart Chatbot", page_icon="🤖")
st.title("🤖 Smart Chatbot")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 100, 500, 250, 50)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ==========================
# VECTOR DB SETUP
# ==========================
client = chromadb.Client()

embedding_function = embedding_functions.OllamaEmbeddingFunction(
    url="http://127.0.0.1:11434/api/embeddings",
    model_name=EMBED_MODEL,
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader("Upload PDF or CSV (optional)", type=["pdf", "csv"])

def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def chunk_text(text, size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_pdf(uploaded_file)
    else:
        text = extract_csv(uploaded_file)

    chunks = chunk_text(text)

    all_docs = collection.get()
    if all_docs["ids"]:
       collection.delete(ids=all_docs["ids"])# clear previous file

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"id_{i}"]
        )

    st.success("File processed. You can now ask about it!")

# ==========================
# CHAT MEMORY
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================
# CHAT INPUT
# ==========================
prompt = st.chat_input("Ask anything...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if document exists
    doc_count = collection.count()

    if doc_count > 0:
        # RAG MODE
        results = collection.query(
            query_texts=[prompt],
            n_results=3
        )

        context = "\n\n".join(results["documents"][0])

        final_prompt = f"""
You are an assistant.

If context is provided, answer ONLY from the context.
If context is empty, answer normally.

Context:
{context}

Question:
{prompt}

Answer clearly and in detail:
"""
    else:
        # NORMAL CHAT MODE
        conversation = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n"

        final_prompt = conversation

    # STREAM RESPONSE
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": final_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 1024
                    }
                },
                stream=True,
                timeout=120
            ) as response:

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        full_response += token
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Error: {str(e)}"
            placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}

    )

