# hil wait for human input
from langgraph.types import interrupt, Command
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
  input: str

def step_1(state: State):
  print("---step 1---")
  pass

def step_2(state: State):
  print("---step 2---")
  pass

def step_3(state: State):
  print("---step 3---")
  pass

builder = StateGraph(State)
builder.add_node(step_1, "step_1")
builder.add_node(step_2, "step_2")
builder.add_node(step_3, "step_3")

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])
graph_png = graph.get_graph().draw_mermaid_png()
with open("interrupt2.png", "wb") as f:
  f.write(graph_png)


initial_input = {"input": "hello world" }

thread = {"configurable": { "thread_id": "23"}}

# 运行 graph, 直到遇到 interrupt
for event in graph.stream(initial_input, thread, stream_mode="values"):
  print(event)


user_approval = input("Do you want to continue? (y/n)")

if user_approval.lower() == "y":
  # 继续运行 graph
  for event in graph.stream(None, thread, stream_mode="values"):
    print(event)

else:
  # 终止 graph
  print("Graph terminated by user.")