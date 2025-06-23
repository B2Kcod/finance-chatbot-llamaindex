from llama_index.core.retrievers import AutoMergingRetriever
from index_manager import vector_index, storage_context

print("Setting up retriever")
vector_retriever = vector_index.as_retriever(similarity_top_k=20)
auto_merging_retriever = AutoMergingRetriever(
    vector_retriever=vector_retriever,
    storage_context=storage_context,
)
print("Retriever set up")