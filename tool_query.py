from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMSingleSelector

import llm_embeddings
from prompts import qa_prompt, summarize_prompt
from retriever import auto_merging_retriever

# --- Question Answering Engine ---
qa_response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    text_qa_template=qa_prompt,
)
qa_query_engine = RetrieverQueryEngine(
    retriever=auto_merging_retriever,
    response_synthesizer=qa_response_synthesizer,
)

# --- Summarization Engine ---
response_synthesizer_summarize = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    summary_template=summarize_prompt
)
summarize_query_engine = RetrieverQueryEngine(
    retriever=auto_merging_retriever,
    response_synthesizer=response_synthesizer_summarize,
)

# --- Router Query Engine ---

qa_tool = QueryEngineTool(
query_engine=qa_query_engine,
metadata=ToolMetadata(
    name="question_answering_tool",
    description=(
        "This tool is exclusively for answering direct questions about financial calculations, formulas, and financial concepts found within the *provided financial documents*. "
        ),
    return_direct=True
),

)
summary_tool = QueryEngineTool(
query_engine=summarize_query_engine,
metadata=ToolMetadata(
    name="summarization_tool",
    description=(
        "This tool is exclusively for summarizing content from the *provided financial documents* when the user's query is not about calculations or formulas. "
    ),
    return_direct=True
),
)