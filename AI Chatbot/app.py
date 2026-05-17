"""
RAG Chatbot — ask questions about your CSV data.
Uses HuggingFace Inference API (free tier) — no payment required.

Setup:
    1. Copy .env.example to .env and add your free HF_TOKEN
       Get one free at: https://huggingface.co/settings/tokens
    2. pip install -r ../requirements.txt
    3. streamlit run app.py

Pre-index CSVs from the repo (optional — you can also upload in the sidebar):
    python ingest.py ../../
"""
import io
import os
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
# It's used to load environment variables from a .env file, such as the HF_TOKEN needed for HuggingFace API access. By calling load_dotenv(), it reads the .env file and sets the environment variables so they can be accessed via os.environ.get() in the code.
from dotenv import load_dotenv 
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "csv_data"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS = 8  # rows retrieved per query

# Free models available on HuggingFace Inference API (no payment needed)
MODELS = {
    "Mistral 7B Instruct (Recommended)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Microsoft Phi-3 Mini (Faster)": "microsoft/Phi-3-mini-4k-instruct",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta",
}

SYSTEM_PROMPT = """You are a knowledgeable data analyst assistant. Answer the user's questions using the retrieved data rows provided with each message.

Guidelines:
- Answer directly and specifically using the provided data.
- If the data is insufficient to answer confidently, say so rather than guessing.
- Format numbers clearly (e.g. use commas: 1,234).
- Use bullet points or short tables when listing multiple items.
- If a calculation is needed, show brief working."""

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CSV Data Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── cached resources ──────────────────────────────────────────────────────────

# Cache the embedding model to avoid reloading it on every interaction. It will only reload if the EMBED_MODEL_NAME changes.
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

# Cache the ChromaDB client to avoid reconnecting on every interaction. The PersistentClient will handle caching of the collection data on disk, so we can just reuse the same client instance.
@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


# Cache the HuggingFace InferenceClient to avoid reloading it on every interaction. It will only reload if the selected model changes or if the HF_TOKEN environment variable changes.
@st.cache_resource
def get_hf_client(model_id: str) -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    return InferenceClient(model=model_id, token=token)


# ── helpers ───────────────────────────────────────────────────────────────────


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(COLLECTION_NAME)


def row_to_text(row: pd.Series, source: str) -> str:
    parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
    return f"[{source}] " + " | ".join(parts)


def ingest_df(df: pd.DataFrame, source: str, collection, embed_model) -> int:
    texts = [row_to_text(row, source) for _, row in df.iterrows()]
    if not texts:
        return 0

    try:
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [f"{source}__{i}" for i in range(len(texts))]
    embeddings = embed_model.encode(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=[{"source": source, "row": i} for i in range(len(texts))],
    )
    return len(texts)


def retrieve(query: str, collection, embed_model, n: int = N_RESULTS) -> list[str]:
    total = collection.count()
    if total == 0:
        return []
    q_emb = embed_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=min(n, total),
    )
    return results["documents"][0] if results["documents"] else []


def stream_response(hf_client: InferenceClient, messages: list[dict]):
    """Yield text chunks from the HuggingFace streaming chat API."""
    stream = hf_client.chat_completion(
        messages=messages,
        stream=True,
        max_tokens=1024,
        temperature=0.3,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── sidebar ───────────────────────────────────────────────────────────────────

embed_model = load_embed_model()
chroma_client = get_chroma_client()
collection = get_or_create_collection(chroma_client)

with st.sidebar:
    st.title("⚙️ Settings")

    # Model selector
    model_label = st.selectbox("LLM Model", list(MODELS.keys()))
    model_id = MODELS[model_label]
    hf_client = get_hf_client(model_id)

    # Token check
    if not os.environ.get("HF_TOKEN"):
        st.warning(
            "**HF_TOKEN not set.**\n\n"
            "Copy `.env.example` → `.env` and add your free token from "
            "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
        )

    st.divider()
    st.subheader("🗂️ Data Sources")

    uploaded = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("📥 Index uploaded files", type="primary", use_container_width=True):
            total_added = 0
            for f in uploaded:
                try:
                    df = pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")))
                    n = ingest_df(df, f.name, collection, embed_model)
                    total_added += n
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if total_added:
                st.success(f"Indexed {total_added:,} rows from {len(uploaded)} file(s).")
                st.rerun()

    st.divider()

    row_count = collection.count()
    if row_count > 0:
        st.metric("Rows in index", f"{row_count:,}")
        try:
            sample = collection.get(limit=500, include=["metadatas"])
            sources = sorted({m["source"] for m in sample["metadatas"]})
            st.caption("Indexed files:")
            for src in sources:
                st.markdown(f"- `{src}`")
        except Exception:
            pass

        if st.button("🗑️ Clear index", use_container_width=True):
            chroma_client.delete_collection(COLLECTION_NAME)
            get_or_create_collection(chroma_client)
            st.session_state.messages = []
            st.rerun()
    else:
        st.info(
            "No data indexed yet.\n\n"
            "Upload CSV files above, or run in the terminal:\n"
            "```\npython ingest.py ../../\n```"
        )

    st.divider()
    st.caption(f"Model: `{model_id}`\nEmbeddings: `{EMBED_MODEL_NAME}`\nVector DB: ChromaDB")

# ── main chat area ────────────────────────────────────────────────────────────

st.title("🤖 CSV Data Chatbot")
st.caption("Powered by HuggingFace (free) + ChromaDB + sentence-transformers")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a question about your data…"):
    if not os.environ.get("HF_TOKEN"):
        st.error("Set your HF_TOKEN in a `.env` file first. See the sidebar for instructions.")
        st.stop()

    if collection.count() == 0:
        st.warning("Please index some CSV files first using the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant rows
    docs = retrieve(user_input, collection, embed_model)
    context_block = "\n".join(f"  {d}" for d in docs)

    # Build messages for the LLM
    # System prompt first, then conversation history, then the augmented user turn
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in st.session_state.messages[:-1]:  # history minus the latest user turn
        llm_messages.append({"role": msg["role"], "content": msg["content"]})

    llm_messages.append({
        "role": "user",
        "content": (
            f"Retrieved data rows (most relevant to my question):\n{context_block}\n\n"
            f"Question: {user_input}"
        ),
    })

    # Stream the response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        try:
            for chunk in stream_response(hf_client, llm_messages):
                reply += chunk
                placeholder.markdown(reply + "▌")
            placeholder.markdown(reply)
        except Exception as e:
            placeholder.error(f"Model error: {e}\n\nTry a different model in the sidebar.")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": reply})
