from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from config import OLLAMA_MODEL, OLLAMA_REQUEST_TIMEOUT, TEMPERATURE, EMBEDDING_MODEL, EMBEDDING_DEVICE, OLLAMA_HOST 

llm = Ollama(
    model=OLLAMA_MODEL,
    temperature=TEMPERATURE,
    request_timeout=OLLAMA_REQUEST_TIMEOUT,
    base_url=OLLAMA_HOST,
)
embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    device=EMBEDDING_DEVICE,
)

Settings.llm = llm
Settings.embed_model = embed_model