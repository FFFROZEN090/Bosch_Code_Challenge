"""CLI script: Build FAISS index from ECU documents."""

import logging

from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.ingest.indexer import build_faiss_index, save_faiss_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading documents...")
    docs = load_all_documents()
    logger.info("Loaded %d documents.", len(docs))

    logger.info("Splitting into chunks...")
    section_chunks, full_doc_chunks = split_all_documents(docs)
    logger.info(
        "Created %d section chunks + %d full-doc chunks.",
        len(section_chunks),
        len(full_doc_chunks),
    )

    logger.info("Building FAISS index from section chunks...")
    index = build_faiss_index(section_chunks)

    save_faiss_index(index)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
