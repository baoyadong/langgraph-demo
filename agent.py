from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from typing_extensions import Literal
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# gpt太卡
# model=ChatOpenAI(model="gpt-4o", temperature=0)
llm=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools }
llm_with_tools = llm.bind_tools(tools)



def llm_call(state: MessagesState):
  """ LLM decides whether to call the tool or not. """
  return {
    "messages": [
      llm_with_tools.invoke(
        [
          SystemMessage(
              content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ] + state["messages"]
      )
    ]
  }

def tool_node(state: dict):
  """ LLM calls the tool and returns the result. """

  result = []
  for tool_call in state["messages"][-1].tool_calls:
     
     print("tool_call", tool_call) #tool_call {'name': 'add', 'args': {'a': 3, 'b': 4}, 'id': 'call_9f13e57b17f846a7add53e', 'type': 'tool_call'}
     tool = tools_by_name[tool_call["name"]]
     overvation = tool.invoke(tool_call["args"])
     result.append(ToolMessage(content = overvation, tool_call_id=tool_call["id"]))
  return {"messages": result}   

def should_continue(state: MessagesState):
  """ LLM decides whether to continue or end the conversation. """
  messages = state["messages"]
  print("state in should_continue", state)
  last_message = messages[-1]
  if last_message.tool_calls:
     return "Action"
  return END


agent_builder = StateGraph(MessagesState)
#add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.set_entry_point("llm_call")
agent_builder.add_conditional_edges(
   "llm_call",
   should_continue,
   {
       "Action": "environment",
       END: END
   }
)
agent_builder.add_edge("environment", "llm_call")

agent = agent_builder.compile()
graph_png = agent.get_graph().draw_mermaid_png()
with open("agent.png", "wb") as f:
  f.write(graph_png)

#invoke agent
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})

for m in messages["messages"]:
   m.pretty_print()
                      
