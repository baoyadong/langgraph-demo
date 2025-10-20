from langchain_openai import ChatOpenAI
import os

# model=ChatOpenAI(model="gpt-4o", temperature=0)
model=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


@tool
def get_weather(location: str) -> str:

    """Use this to get weather information."""
    if any([city in location.lower() for city in ["nyc", "new york city"]]):
        return "It might be cloudy in nyc, with a chance of rain and temperatures up to 80 degrees."
    elif any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's always sunny in sf"
    else:
        return f"I am not sure what the weather is in {location}"

tools = [get_weather]

memory = MemorySaver()
# 创建react代理
graph = create_react_agent(
   model, tools=tools, interrupt_before=["tools"],checkpointer=memory
)


def print_stream(stream):
  for s in stream:
     message = s["messages"][-1]
     if isinstance(message, tuple):
       print(message)
     else:
       message.pretty_print()
# 配置参数
config = {
   "configurable": { "thread_id": "42"}
}
# inputs = {"messages": [{ "user", "what is the weather in nyc?"}]}
inputs = {"messages": [HumanMessage("what is the weather in nyc?")]}
print_stream(graph.stream(inputs, config=config, stream_mode="values"))

#获取图的快照
snapshot = graph.get_state(config)

#打印下一步信息
print("Next step: ", snapshot.next)


#打印后续流输出
print_stream(graph.stream(None, config=config, stream_mode="values"))

graph_png = graph.get_graph().draw_mermaid_png()
with open("create_react_agent_hitl.png", "wb") as f:
  f.write(graph_png)
