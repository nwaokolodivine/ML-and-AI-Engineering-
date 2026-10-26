# Document RAG Assistant

A small Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, OpenAI embeddings, and an OpenAI language model.

## Features

- Upload PDF and TXT documents.
- Extract document text and split it into overlapping chunks.
- Create vector embeddings for each chunk.
- Retrieve relevant chunks using cosine similarity.
- Generate answers grounded in retrieved context.
- Display source chunks and similarity scores for transparency.
- Configure chunk size, overlap, top-k retrieval, and model.

## Architecture

1. **Ingestion:** PDF/TXT upload and text extraction.
2. **Chunking:** split extracted text into overlapping chunks.
3. **Embedding:** convert chunks into vectors with `text-embedding-3-small`.
4. **Retrieval:** embed the question and rank chunks by cosine similarity.
5. **Generation:** pass the retrieved context to an LLM and ask it to answer with source labels.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with API access and billing configured
- VS Code (recommended)

## Run locally

### 1. Open this folder in VS Code

Open the folder containing `app.py`.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Set your API key

Copy `.env.example` to `.env`, then put your own API key in `.env`.

Important: this starter app reads environment variables; it does not automatically load `.env`. In PowerShell, set the variable for the current terminal session:

```powershell
$env:OPENAI_API_KEY="your-real-api-key"
```

Do not paste your API key into `app.py`, commit it, or share it publicly.

### 5. Start the app

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Try it

1. Upload a text-based PDF or `.txt` file.
2. Click **Process documents**.
3. Ask a question whose answer appears in the uploaded content.
4. Open **View retrieved source chunks** to inspect the evidence used.

## Known limitations

- Scanned PDFs may need OCR; this version extracts embedded PDF text only.
- Embeddings are held in memory and are recreated after restart.
- Chunking is character-based, not token-based.
- Similarity retrieval is basic cosine similarity; this starter version does not include reranking.
- The model can still make mistakes. Verify important answers against the original documents.
- API usage may incur charges.

## Potential improvements

- Persistent vector storage (FAISS, Chroma, or a hosted vector database).
- OCR support for scanned PDFs.
- Token-aware chunking and metadata filters.
- Retrieval evaluation and answer-quality tests.
- Reranking and hybrid keyword/vector search.
- Authentication, file-size limits, and deployment.

## GitHub checklist

Before pushing:
- Confirm the app runs locally.
- Test it with a document you are allowed to use.
- Keep `.env` out of Git.
- Add screenshots and a short demo GIF if possible.
- Describe what you implemented and what limitations remain.

## CV note

Only list this as a completed project after you have run and tested it. A truthful project description could be:

**Document RAG Assistant | Python, Streamlit, OpenAI API, Embeddings, Semantic Search**
- Built a document-question-answering app that ingests PDF/TXT files, chunks extracted text, creates embeddings, retrieves relevant passages by cosine similarity, and generates grounded answers with source labels.
- Added configurable retrieval settings and an interface for inspecting retrieved evidence.
