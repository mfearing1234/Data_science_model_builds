# RAG Data Chatbot — Ask Questions About Your CSV Files

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload CSV data and ask questions about it in plain English. Relevant rows are retrieved from a vector database and passed to an open-source LLM, so answers are grounded in your actual data rather than the model's general knowledge.

Runs entirely on free tooling — no paid API needed.

## How it works

```
CSV rows ──► text ("col: value | col: value") ──► embeddings (all-MiniLM-L6-v2) ──► ChromaDB
                                                                                       │
User question ──► embedding ──► top-8 nearest rows ◄───────────────────────────────────┘
                                         │
                                         ▼
               System prompt + chat history + retrieved rows ──► Qwen 2.5 LLM (Hugging Face)
                                                                       │
                                                                       ▼
                                                         Streamed answer in Streamlit
```

1. **Ingest** — each CSV row is serialized to text and embedded with `sentence-transformers`.
2. **Store** — embeddings and source metadata (file name, row number) are saved to a persistent **ChromaDB** collection. Re-indexing a file replaces its old rows.
3. **Retrieve** — the user's question is embedded and the most similar rows are pulled back.
4. **Generate** — retrieved rows plus conversation history go to a **Qwen 2.5** instruct model through the Hugging Face Inference API. The system prompt instructs the model to cite rows, show calculations, and say so when the data is insufficient instead of guessing.

## Features

- Upload and index CSV files directly from the sidebar, or bulk-index from the command line
- Choose between four Qwen 2.5 models (3B, 7B, 72B, and Coder 7B) to trade speed for quality
- Streaming responses with multi-turn chat memory
- Sidebar shows indexed row count and source files; one-click index reset

## Tech stack

`Streamlit` · `ChromaDB` · `sentence-transformers` · `Hugging Face Inference API` · `Qwen 2.5` · `pandas` · `python-dotenv`

## How to run

1. Get a free Hugging Face token at <https://huggingface.co/settings/tokens>.
2. Set up your environment:

   ```bash
   pip install -r ../requirements.txt
   cp .env.example .env      # then paste your token after HF_TOKEN=
   ```

3. (Optional) Pre-index CSV files from the command line:

   ```bash
   python ingest.py ../                  # every CSV in the repo
   python ingest.py path/to/file.csv     # a single file
   python ingest.py --clear              # wipe the index
   ```

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit chat UI, retrieval, and LLM streaming |
| `ingest.py` | Command-line CSV indexer |
| `.env.example` | Template for the `HF_TOKEN` environment variable |

The `chroma_db/` vector store is created locally on first run and is not committed.
