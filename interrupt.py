from langgraph.types import interrupt, Command
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph


class State(TypedDict):
    some_text: str

def human_node(state: State):
    value = interrupt(
        {
            "text_to_revise": state["some_text"]
        }
    )
    return {
        "some_text": value
    }

graph_builder = StateGraph(State)
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer)
config = {"configurable": {"thread_id": 12}}

result = graph.invoke({"some_text": "original text"}, config=config)
print(result)

# resume from the last checkpoint
print(graph.invoke(Command(resume="Edited text"), config=config))
