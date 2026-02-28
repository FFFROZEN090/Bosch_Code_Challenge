"""Build and persist FAISS vector index from document chunks."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from me_assistant.config import EMBEDDING_MODEL, FAISS_INDEX_DIR

logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model instance used across the project."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_faiss_index(chunks: list[Document]) -> FAISS:
    """Build a FAISS index from a list of LangChain Document chunks.

    Args:
        chunks: Section-level chunks with metadata (from splitter).

    Returns:
        A FAISS vector store ready for similarity search.
    """
    embeddings = get_embeddings()
    logger.info("Building FAISS index from %d chunks...", len(chunks))
    index = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully.")
    return index


def save_faiss_index(index: FAISS) -> None:
    """Save FAISS index to disk."""
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(FAISS_INDEX_DIR))
    logger.info("FAISS index saved to %s", FAISS_INDEX_DIR)


def load_faiss_index() -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    index = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from %s", FAISS_INDEX_DIR)
    return index
