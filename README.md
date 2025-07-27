Financial Expert Chatbot

This project implements a financial expert chatbot powered by LlamaIndex, leveraging a local Ollama LLM and HuggingFace embeddings for efficient and accurate responses based on your provided financial documents.

*Evaluations not integrated in this version*


Features

Intelligent Q&A: 		            Answers direct questions about financial calculations, formulas, and concepts.
Content Summarization: 		      Summarizes financial content from documents.
Hierarchical Document 		      Processing: Uses a hierarchical node parser for effective document chunking and retrieval.
Auto-Merging Retrieval: 	      Employs AutoMergingRetriever for improved context retrieval.
Local LLM Integration: 		      Utilizes Ollama to run large language models locally.
HuggingFace Embeddings: 	      Uses BAAI/bge-small-en-v1.5 for generating document embeddings.
Conversational Agent: 		      Built with ReActAgent for dynamic tool selection (Q&amp;A or summarization) based on user queries.
Evaluation Dashboard (TruLens): Includes setup for a TruLens dashboard to evaluate model performance (groundedness, answer relevance, context relevance).


System Requirements

*MUST HAVE NVIDIA GPU* ( or select EMBEDDING_DEVICE = "cpu" ) in config.py

Operating System:  Windows
Processor:         Intel Core i5-9300H 2.40GHz
RAM:               16 GB
GPU:               NVIDIA GeForce GTX 1660 Ti with 6 GB VRAM
Storage:           500 GB SSD

Setup Instructions:

1. Install Cuda Toolkit 12.8 
2. Visual Studio Code with Python 3.11 installed
3. Ollama
	After installing ollama open a powershell terminal and run the following commands:
	ollama run lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M
	ollama run gemma2:2b

4. pip install -r requirements.txt
5. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

Notes: 

Cuda 12.8 version must match  --extra-index-url ..../cu128

Make sure ollama is up and running

The first time you run agent.py (or if the ./index directory does not exist), the system will automatically download bge small 1.5 and load your PDF documents from the data folder, parse them into nodes, generate embeddings, and store the index. This process can take some time depending on the size of your documents.
If you add new documents or want to rebuild the index, delete the ./index directory AND replace the pdfs in the ./data folder with your own


Usage:

Bash

python agent.py

You will be prompted to enter your queries. Type quit to exit the chatbot.
