import os

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
SIMILARITY_THRESHOLD = 0.45
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-70b-8192"

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")
