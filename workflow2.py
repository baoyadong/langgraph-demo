# parallel processing with langchain-openai

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import os

llm=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

class State(TypedDict):
  topic:str
  joke: str
  story:str
  poem: str
  combined_output:str

# joke nodes
def call_llm_1(state: State):
  """First LLM call to generate initial joke"""
  msg = llm.invoke(f"Write a joke about {state['topic']}")
  return {"joke": msg.content}

# story nodes
def call_llm_2(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

# poem nodes
def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    """Combines all outputs into a single string"""
    combines = f"Here is a story, joke and poem about {state['topic']}!\n\n"
    combines += f"Story: {state['story']}\n\n"
    combines += f"Joke: {state['joke']}\n\n"
    combines += f"Poem: {state['poem']}"
    return {"combined_output": combines}

# create state graph
parallel_graph = StateGraph(State)
# add nodes to graph
parallel_graph.add_node("call_llm_1", call_llm_1)
parallel_graph.add_node("call_llm_2", call_llm_2)
parallel_graph.add_node("call_llm_3", call_llm_3)
parallel_graph.add_node("aggregator", aggregator)

#add edges to connect nodes
parallel_graph.add_edge(START, "call_llm_1")
parallel_graph.add_edge(START, "call_llm_2")
parallel_graph.add_edge(START, "call_llm_3")
parallel_graph.add_edge("call_llm_1", "aggregator")
parallel_graph.add_edge("call_llm_2", "aggregator")
parallel_graph.add_edge("call_llm_3", "aggregator")
parallel_graph.add_edge("aggregator", END)
parallel_chain = parallel_graph.compile()

graph_png = parallel_chain.get_graph().draw_mermaid_png()
with open("workflow2.png", "wb") as f:
  f.write(graph_png)


# invoke chain
state = parallel_chain.invoke({"topic": "书本"})
print(state["combined_output"])