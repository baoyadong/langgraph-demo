# 评估器

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, Annotated
import operator
from langgraph.types import Send
import os

llm=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

#Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
      description="Whether the joke is funny or not"
    )
    feedback: str = Field(
        description="if the joke is not funny, the feedback will be provided to improve the joke"
    )

evaluator = llm.with_structured_output(Feedback)

#Nodes
def llm_call_generaotr(state: State):
     """ llm generates a joke based on the given topic """
     if state.get("feedback"):
         msg = llm.invoke(
             f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
         )
     else:
         msg = llm.invoke(f"Write a joke about {state['topic']}")

     return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """ llm evaluates the joke and returns a grade and feedback """
    grade = evaluator.invoke(f"Grade the joke: {state['joke']}, and return grade: funny or not funny. and return feedback to improve the joke. Response in json format.")
    print(grade, "llm_call_evaluator")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback }

# conditional edge function to route back to joke generator or end based upon the feedback
def route_joke(state: State):
    """ routes back to joke generator or end based upon the feedback """
    if state.get("funny_or_not") == "funny":
        return "Accepted"
    elif state.get("funny_or_not") == "not funny":
        return "Rejected + Feedback"

optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generaotr", llm_call_generaotr)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generaotr")
optimizer_builder.add_edge("llm_call_generaotr", "llm_call_evaluator")

optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generaotr"
    }
)

optimizer = optimizer_builder.compile()
graph_png = optimizer.get_graph().draw_mermaid_png()
with open("workflow5.png", "wb") as f:
  f.write(graph_png)

state = optimizer.invoke({"topic": "月球旅行"})
print(state["joke"])

