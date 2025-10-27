# agent with rag
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


