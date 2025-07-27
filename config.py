# Directory to persist the LlamaIndex index
PERSIST_DIR = "./index"

# Ollama LLM model details
OLLAMA_MODEL = "llama3:8b-instruct-q4_K_M"
OLLAMA_REQUEST_TIMEOUT = 3600.0
TEMPERATURE = 0

# HuggingFace Embedding model details
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # maximum 512 input
EMBEDDING_DEVICE = "cuda"
CHUNK_SIZES = [512, 256, 128] # only the last element will be embedded

# Directory containing your PDF documents
DATA_DIR = "./data"

