# Configuration for Company Knowledge Base Assistant

# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = "./docs"

# Chunking configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index paths (relative to src directory)
FAISS_INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"
FTS_INDEX_PATH = "index.fts.sqlite"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:0.6b"

# RAG retrieval configuration
TOP_K = 5
