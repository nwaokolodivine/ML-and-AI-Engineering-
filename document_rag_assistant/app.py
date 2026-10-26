import os
from typing import List, Dict

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="Document RAG Assistant", page_icon="📚", layout="wide")

st.title("📚 Document RAG Assistant")
st.caption("Ask questions about your PDFs and text files using Retrieval-Augmented Generation (RAG).")

st.markdown(
    """
    Upload one or more PDF/TXT files. The app extracts the text, splits it into chunks,
    creates embeddings, retrieves the most relevant chunks for each question, and asks an
    LLM to answer using that context.
    """
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Add it to a .env file or your environment, then restart the app.")
    st.code('OPENAI_API_KEY="your-api-key-here"', language="bash")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Settings")
    embedding_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-small"],
        help="Converts text into vectors for semantic retrieval."
    )
    llm_model = st.selectbox(
        "LLM",
        ["gpt-4.1-mini", "gpt-4.1"],
        index=0,
        help="The model used to generate answers from retrieved context."
    )
    chunk_size = st.slider("Chunk size (characters)", 500, 1800, 1000, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)
    top_k = st.slider("Retrieved chunks per question", 1, 8, 4)

if overlap >= chunk_size:
    st.warning("Chunk overlap should be smaller than chunk size.")
    st.stop()


def extract_text(uploaded_file) -> str:
    """Extract text from a PDF or plain-text upload."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {page_number}]\n{text}")
        return "\n\n".join(pages)
    if name.endswith(".txt"):
        raw = uploaded_file.getvalue()
        return raw.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def split_into_chunks(text: str, size: int, overlap_chars: int) -> List[str]:
    """Simple character-based chunking with overlap."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    step = size - overlap_chars
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def embed_texts(texts: List[str], model: str) -> np.ndarray:
    """Create embeddings in batches to avoid overly large requests."""
    vectors = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in response.data])
    matrix = np.array(vectors, dtype=np.float32)
    # Normalize vectors so dot product equals cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def build_index(uploaded_files, size: int, overlap_chars: int, model: str) -> List[Dict]:
    """Extract documents, chunk text, and create an in-memory embedding index."""
    records = []
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        if not text.strip():
            st.warning(f"No extractable text found in {uploaded_file.name}. Scanned PDFs may need OCR.")
            continue
        chunks = split_into_chunks(text, size, overlap_chars)
        for index, chunk in enumerate(chunks, start=1):
            records.append({
                "source": uploaded_file.name,
                "chunk_id": index,
                "text": chunk,
            })

    if not records:
        return []

    embeddings = embed_texts([record["text"] for record in records], model)
    for record, vector in zip(records, embeddings):
        record["embedding"] = vector
    return records


def retrieve(question: str, records: List[Dict], model: str, count: int) -> List[Dict]:
    """Retrieve the top-k chunks with the highest cosine similarity."""
    query_vector = embed_texts([question], model)[0]
    scores = np.array([float(np.dot(record["embedding"], query_vector)) for record in records])
    top_indices = np.argsort(scores)[::-1][:count]
    results = []
    for idx in top_indices:
        item = dict(records[int(idx)])
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results


def answer_question(question: str, retrieved: List[Dict], model: str) -> str:
    """Generate a grounded answer and instruct the model not to invent missing facts."""
    context_parts = []
    for i, item in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i}: {item['source']}, chunk {item['chunk_id']}]\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    response = client.responses.create(
        model=model,
        instructions=(
            "You are a careful document question-answering assistant. "
            "Answer using only the supplied document context. "
            "If the context does not contain the answer, say you could not find enough information "
            "in the uploaded documents. Do not invent facts. Cite sources inline using labels like "
            "[Source 1] that correspond to the supplied context."
        ),
        input=f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
    )
    return response.output_text


uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Text-based PDFs and .txt files are supported. Scanned/image-only PDFs may not contain extractable text."
)

if "records" not in st.session_state:
    st.session_state.records = []
if "indexed_fingerprint" not in st.session_state:
    st.session_state.indexed_fingerprint = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    fingerprint = tuple((f.name, len(f.getvalue())) for f in uploaded_files)
    if st.button("Process documents", type="primary"):
        with st.spinner("Extracting text and creating embeddings..."):
            try:
                st.session_state.records = build_index(uploaded_files, chunk_size, overlap, embedding_model)
                st.session_state.indexed_fingerprint = fingerprint
                st.session_state.chat_history = []
            except Exception as exc:
                st.error(f"Could not process documents: {exc}")
                st.stop()

if st.session_state.records:
    st.success(f"Ready: indexed {len(st.session_state.records)} text chunks.")
    question = st.text_input("Ask a question about your documents", placeholder="What are the main findings?")
    ask = st.button("Ask question")
    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    retrieved = retrieve(question, st.session_state.records, embedding_model, top_k)
                    answer = answer_question(question, retrieved, llm_model)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": retrieved,
                    })
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")

    for turn in reversed(st.session_state.chat_history):
        st.markdown("---")
        st.markdown(f"**Question:** {turn['question']}")
        st.markdown(turn["answer"])
        with st.expander("View retrieved source chunks"):
            for item in turn["sources"]:
                st.markdown(
                    f"**{item['source']} — chunk {item['chunk_id']} "
                    f"(similarity: {item['score']:.3f})**"
                )
                st.write(item["text"])
else:
    st.info("Upload PDF/TXT documents and select **Process documents** to get started.")

st.markdown("---")
st.caption("Demo project: document retrieval is in-memory and resets when the app restarts. Do not upload confidential documents.")
