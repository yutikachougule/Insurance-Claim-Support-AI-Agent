"""
ingest.py - Build a FAISS vector store from the insurance documents in data/

Usage:
    python ingest.py

This will:
1. Load all .md files from the data/ directory
2. Split them into overlapping chunks
3. Embed the chunks using a local HuggingFace embedding model (all-MiniLM-L6-v2)
4. Save the resulting FAISS index to vector_store/

"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir: str):
    """Load every .md file in data_dir as a LangChain Document."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from '{data_dir}/'")
    return docs


def split_documents(docs):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks):
    """Embed chunks with a local HuggingFace model and build a FAISS index."""

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads ~80MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def main():
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)

    # Sanity check: print the first chunk from each source file
    seen_sources = set()
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "unknown")).name
        if source not in seen_sources:
            seen_sources.add(source)
            preview = chunk.page_content[:200].replace("\n", " ")
            print(f"\n[{source}] first chunk preview:\n  {preview}...")

    vector_store = build_vector_store(chunks)

    Path(VECTOR_STORE_DIR).mkdir(exist_ok=True)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"\nSaved FAISS index to '{VECTOR_STORE_DIR}/' ({len(chunks)} vectors)")


if __name__ == "__main__":
    main()