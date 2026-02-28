"""Load ECU markdown documents with metadata."""

from pathlib import Path
from typing import NamedTuple

from me_assistant.config import DOCS_DIR, DOC_METADATA


class LoadedDocument(NamedTuple):
    """A loaded document with its content and metadata."""
    content: str
    metadata: dict


def _fix_malformed_table(text: str) -> str:
    """Fix markdown table rows that are missing the leading pipe character.

    ECU-700_Series_Manual.md line 18 has a malformed row:
        **CAN Interface**       | Single Channel ...
    Should be:
        | **CAN Interface**       | Single Channel ...
    """
    lines = text.split("\n")
    fixed = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # Detect table start: line begins with |
        if stripped.startswith("|"):
            in_table = True
            fixed.append(line)
        elif in_table and "|" in stripped and not stripped.startswith("#"):
            # Line is inside a table block but missing leading pipe
            fixed.append("| " + stripped)
        else:
            if in_table and not stripped:
                in_table = False
            fixed.append(line)
    return "\n".join(fixed)


def load_document(filepath: Path) -> LoadedDocument:
    """Load a single markdown document and attach metadata.

    Args:
        filepath: Path to the markdown file.

    Returns:
        LoadedDocument with cleaned content and metadata dict containing
        series, model, and source_file fields.
    """
    filename = filepath.name

    if filename not in DOC_METADATA:
        raise ValueError(
            f"Unknown document: {filename}. "
            f"Expected one of: {list(DOC_METADATA.keys())}"
        )

    raw_text = filepath.read_text(encoding="utf-8")
    content = _fix_malformed_table(raw_text)

    metadata = {
        **DOC_METADATA[filename],
        "source_file": filename,
    }

    return LoadedDocument(content=content, metadata=metadata)


def load_all_documents() -> list[LoadedDocument]:
    """Load all ECU documents from the docs directory.

    Returns:
        List of LoadedDocument, one per markdown file.
    """
    docs = []
    for filename in sorted(DOC_METADATA.keys()):
        filepath = DOCS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")
        docs.append(load_document(filepath))
    return docs
