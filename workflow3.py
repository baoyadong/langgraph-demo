from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal
import os

llm=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

class Route(BaseModel):
   step: Literal["poem", "story", "joke"] = Field(
      None,
      description = "The next step in the routing process"
   )

#Argument the LLM with the schema for structured output
router = llm.with_structured_output(Route)

class State(TypedDict):
   input: str
   decision: str
   output: str

#Nodes
def llm_call_1(state: State):
   """write a story"""
   result = llm.invoke(state["input"])
   return {"output": result.content }

def llm_call_2(state: State):
   """write a poem"""
   result = llm.invoke(state["input"])
   return {"output": result.content }

def llm_call_3(state: State):
   """tell a joke"""
   result = llm.invoke(state["input"])
   return {"output": result.content }
  

def llm_call_router(state: State):
   """router the input to the appropriate node"""
   # run the augmented llm with structed output to server as routing logic
   decision = router.invoke(
      [
         SystemMessage(
            content = "Route the input to story, poem or joke based on the user's request."  + 
                       "you must response with json, 返回值要包含step字段，值为story, poem, joke"
         ),
         HumanMessage(
            content = state["input"]
         )
      ]
   )
   return {"decision": decision.step}

def route_decision(state: State):
   """decide the next node based on the routing decision"""
   if state["decision"] == "story":
      return "llm_call_1"
   elif state["decision"] == "poem":
      return "llm_call_2"
   elif state["decision"] == "joke":
      return "llm_call_3"
   
#build workflow graph
router_builder = StateGraph(State)

#add nodes to the graph
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

#add edges to the graph
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
   "llm_call_router",
   route_decision,
   {
      "llm_call_1": "llm_call_1",
      "llm_call_2": "llm_call_2",
      "llm_call_3": "llm_call_3"
   }
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

#compile the graph to a chain
router_workflow = router_builder.compile()
graph_png = router_workflow.get_graph().draw_mermaid_png()

with open("workflow3.png", "wb") as f:
  f.write(graph_png)

state = router_workflow.invoke({"input": "Write me a poem about 天气"})

print(state["output"])
