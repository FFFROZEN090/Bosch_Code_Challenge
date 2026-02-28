"""Central configuration for ME Engineering Assistant."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
TEST_QUESTIONS_PATH = DATA_DIR / "test-questions.csv"

# Document metadata mapping: filename -> (series, model)
DOC_METADATA = {
    "ECU-700_Series_Manual.md": {"series": "700", "model": "ECU-750"},
    "ECU-800_Series_Base.md": {"series": "800", "model": "ECU-850"},
    "ECU-800_Series_Plus.md": {"series": "800", "model": "ECU-850b"},
}

# LLM configuration
LLM_PROVIDER = "ollama"  # "ollama" | "openai"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval configuration
FAISS_TOP_K = 5
