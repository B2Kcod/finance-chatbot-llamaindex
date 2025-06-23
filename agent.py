from llama_index.core.agent import ReActAgent
from llama_index.core.memory import Memory
from tool_query import qa_tool, summary_tool
from llm_embeddings import llm
from prompts import agent_prompt

memory = Memory.from_defaults(token_limit=4000)


agent = ReActAgent.from_tools(
    [qa_tool,summary_tool], 
    llm=llm, 
    memory=memory,
    verbose=True
)

agent.update_prompts({"agent_worker:system_prompt": agent_prompt})

while True:
    query = input("-Enter your query (type quit to exit): ")
    response = agent.chat(query)
    print(response)
    if query == "quit":
        break

