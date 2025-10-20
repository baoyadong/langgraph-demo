from pydantic import BaseModel, Field
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
   topic: str
   joke: str
   improved_joke: str
   final_joke: str

#node
def generate_joke(state: State):
    """ first LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return { "joke": msg.content}

def check_punchline(state: State):
    """ check if the joke has a punchline"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail" 

def improve_joke(state: State):
    """ second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return { "improved_joke": msg.content}

def polish_joke(state: State):
    """ third LLM call to polish the joke"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return { "final_joke": msg.content}

#graph
workflow = StateGraph(State)

workflow.add_node("generate_joke", generate_joke)

workflow.add_node("improve_joke", improve_joke)

workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges("generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END})
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()
graph_png = chain.get_graph().draw_mermaid_png()
with open("workflow1.png", "wb") as f:
  f.write(graph_png)

state = chain.invoke({"topic": "书本"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Joke failed quality gate - no punchline detected!")
