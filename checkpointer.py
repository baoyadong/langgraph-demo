from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
   foo: str
   bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer = checkpointer)
graph_png = graph.get_graph().draw_mermaid_png()
with open("checkpointer.png", "wb") as f:
  f.write(graph_png)

config = {"configurable": {"thread_id": "100"}}
graph.invoke({"foo": ""}, config)

print(graph.get_state(config))

print("========")
# 获取历史状态
print(list(graph.get_state_history(config)))


