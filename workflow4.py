# 编排器
# 此工作流非常适合那些无法预测所需子任务的复杂任务

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

class Section(BaseModel):
    name: str = Field(
        description = "Name for the section of the report"
    )
    description: str = Field(
        description = "Brief overview of the main topics and concepts to be covered in this section."
    )
class Sections(BaseModel):
    sections: list[Section] = Field(
        description = "List of sections in the report"
    )

planner = llm.with_structured_output(Sections)

# Graph state
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

#Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


#Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report based on the given state."""
    report_sections = planner.invoke(
        [
            SystemMessage(content=f"Generate a plan for the report. you must response with json. The report should be made of List of section. the section should have name and description."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}")
        ]
    )
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """ worker writes a section of the report"""
    section = llm.invoke(
        [
            SystemMessage(
              content=f"Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting. " + 
              "you must response with json"
            ),
            HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}")
        ]
    )
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """Synthesizer that combines all the sections into a final report."""
    completed_sections = state['completed_sections']
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}

#Send, 它允许您动态创建工作器节点并将特定输入发送给每个节点;
def assign_worker(state: State):
    """Assigns a worker to each section of the report."""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

#Graph
orchestrator_worker_builder = StateGraph(State)
# Add nodes to the graph
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)


# Add edges to the graph

orchestrator_worker_builder.add_edge(START, "orchestrator")

orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_worker, ["llm_call"])

orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)


orchestrator_worker = orchestrator_worker_builder.compile()
graph_png = orchestrator_worker.get_graph().draw_mermaid_png()
with open("workflow4.png", "wb") as f:
  f.write(graph_png)


state = orchestrator_worker.invoke({"topic": "Create a report on 星际旅行"})

print(state["final_report"])

