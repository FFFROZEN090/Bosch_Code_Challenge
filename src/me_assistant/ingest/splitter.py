"""Split ECU documents into chunks with metadata for vector indexing."""

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from me_assistant.ingest.loader import LoadedDocument


# Headers to split on — covers the heading levels used in our ECU docs
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Pattern: lines like "**1. Introduction**" or "**2. Technical Specifications: ECU-750**"
# These are bold-numbered section titles used in ECU-700 instead of ## headers.
_BOLD_SECTION_RE = re.compile(r"^\*\*(\d+)\.\s+(.+?)\*\*\s*$")


def _normalize_bold_headers(text: str) -> str:
    """Convert bold-numbered section titles to standard markdown ## headers.

    ECU-700 uses "**1. Introduction**" instead of "## Introduction".
    This normalizes them so MarkdownHeaderTextSplitter can split properly.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        match = _BOLD_SECTION_RE.match(line.strip())
        if match:
            title = match.group(2)
            result.append(f"## {title}")
        else:
            result.append(line)
    return "\n".join(result)


def _build_section_path(header_metadata: dict) -> list[str]:
    """Extract ordered section path from splitter header metadata.

    The MarkdownHeaderTextSplitter returns metadata like
    {"h1": "Title", "h2": "Section"}.  We convert this to a list:
    ["Title", "Section"].
    """
    path = []
    for key in ("h1", "h2", "h3"):
        if key in header_metadata:
            path.append(header_metadata[key])
    return path


def split_document(loaded_doc: LoadedDocument) -> list[Document]:
    """Split a single loaded document into LangChain Document chunks.

    Strategy:
    - Split by markdown headers to preserve section semantics.
    - Each section becomes one chunk (docs are small, ~30 lines each).
    - Tables are kept intact within their section — no further splitting.
    - Every chunk inherits the document-level metadata (series, model, source_file)
      plus a section_path derived from headers.

    Args:
        loaded_doc: A LoadedDocument from the loader.

    Returns:
        List of LangChain Document objects, each with enriched metadata.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    normalized_content = _normalize_bold_headers(loaded_doc.content)
    sections = splitter.split_text(normalized_content)

    chunks = []
    for idx, section in enumerate(sections):
        section_path = _build_section_path(section.metadata)

        chunk_metadata = {
            **loaded_doc.metadata,
            "section_path": section_path,
            "chunk_id": f"{loaded_doc.metadata['source_file']}:{idx}",
            "chunk_type": "section",
        }

        chunks.append(Document(
            page_content=section.page_content,
            metadata=chunk_metadata,
        ))

    return chunks


def create_full_doc_chunk(loaded_doc: LoadedDocument) -> Document:
    """Create a single chunk containing the full document text.

    Used by the COMPARE retrieval path, which injects all documents
    as context rather than relying on similarity search.

    Args:
        loaded_doc: A LoadedDocument from the loader.

    Returns:
        A single LangChain Document with the complete text.
    """
    return Document(
        page_content=loaded_doc.content,
        metadata={
            **loaded_doc.metadata,
            "section_path": [],
            "chunk_id": f"{loaded_doc.metadata['source_file']}:full",
            "chunk_type": "full_document",
        },
    )


def split_all_documents(
    loaded_docs: list[LoadedDocument],
) -> tuple[list[Document], list[Document]]:
    """Split all loaded documents into section chunks and full-doc chunks.

    Args:
        loaded_docs: List of LoadedDocument from load_all_documents().

    Returns:
        Tuple of (section_chunks, full_doc_chunks).
        - section_chunks: for FAISS indexing (single-source retrieval)
        - full_doc_chunks: for COMPARE retrieval (all-docs injection)
    """
    section_chunks = []
    full_doc_chunks = []

    for doc in loaded_docs:
        section_chunks.extend(split_document(doc))
        full_doc_chunks.append(create_full_doc_chunk(doc))

    return section_chunks, full_doc_chunks
