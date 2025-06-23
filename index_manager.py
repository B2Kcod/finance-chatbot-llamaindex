from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.readers.file import PyMuPDFReader
import os
from config import PERSIST_DIR, DATA_DIR, CHUNK_SIZES
from llm_embeddings import llm, embed_model 


# Loads an existing LlamaIndex from disk or creates a new one if it doesn't exist
# Note: Delete the index folder in order to create new embeddings from other books!

# If there is not a directory called index we create storage for the retrieval
if not os.path.exists(PERSIST_DIR):
    print("Creating new index...")
    # Loading the pdf files from "data" folder
    print("Loading data")
    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        file_extractor={".pdf": PyMuPDFReader()}
    ).load_data()
    print("Data loaded")
    print("Creating nodes")
    # Splitting the documents into nodes for automerging retrieval
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    print("Nodes created")
    # Setting the global setting of the node parser (later used for automerging retrieval)
    Settings.node_parser = node_parser
    print("Storing nodes")
    # Storing the nodes into a BaseDocument store
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    print("Nodes Stored")
    print("Creating leaf nodes vectorstore index")
    # Creating a vectorstore of leaf nodes and their embeddings (default it takes the Settings.embed_model)
    vector_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )
    # Saves the storage of the root nodes and leaf nodes with embeddings in the index directory
    storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index created and persisted.")
# If there is a index folder, we load it (Make sure to delete the index folder when you want to update data)
else:
    print("Loading existing index")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    vector_index = load_index_from_storage(storage_context=storage_context)
    # Re-set the node parser for existing index as it's part of Settings
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    Settings.node_parser = node_parser
    print("Index loaded.")
    