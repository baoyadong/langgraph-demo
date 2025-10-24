from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
  age: int

def get_valid_age(state: State) -> State:
  prompt = "Plase enter your age: (must be a non-negative integer)."
  while True:
    user_input = interrupt(prompt)
    try:
      age = int(user_input)
      if age < 0:
        raise ValueError("Age must be non-negative.")
      break
    except ValueError:
      prompt = "Invalid input. Please try again."
  return {"age": age}

def report_age(state: State) ->State:
  print(f"Your age is {state['age']}.")
  return state


builder = StateGraph(State)
builder.add_node("get_valid_age", get_valid_age)
builder.add_node("report_age", report_age)

builder.set_entry_point("get_valid_age")
builder.add_edge("get_valid_age", "report_age")
builder.add_edge("report_age", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
graph_png = graph.get_graph().draw_mermaid_png()
with open("interrupt4.png", "wb") as f:
  f.write(graph_png)

#config
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({}, config=config)

print(result)

result = graph.invoke(Command(resume="not a number"), config=config)
print(result)

final_result = graph.invoke(Command(resume="25"), config=config)
print(final_result)  # Should include the valid age
