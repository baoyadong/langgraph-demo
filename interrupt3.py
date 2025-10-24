
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
  input: str
  user_feedback: str


def step_1(state: State):
  print("step 1")
  pass

def human_feedback(state: State):
  print("human feedback")
  # feedback的值是从Command中的resume获取的
  feedback = interrupt("Please provide feedback of the user: ")
  print(f"User feedback: {feedback}")
  return {"user_feedback": feedback}

def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)


memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
graph_png = graph.get_graph().draw_mermaid_png()
with open("interrupt3.png", "wb") as f:
  f.write(graph_png)

# Input
initial_input = {"input": "hello world"}
thread = {"configurable": {"thread_id": "1"}}
# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="updates"):
  print(event)
  print("\n")

user_approval = input("Do you want to continue? (y/n)")

if user_approval.lower() == "y":
  for event in graph.stream(
      Command(resume="go to step 3!"),
      thread,
      stream_mode="updates",
  ):
      print(event)
      print("\n")