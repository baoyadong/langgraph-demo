from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
import os
from langgraph.prebuilt import create_react_agent
from utils import pretty_print_messages
from langgraph_supervisor import create_supervisor



# gpt太卡
# model=ChatOpenAI(model="gpt-4o", temperature=0)
model=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

web_search = TavilySearch(max_results=3)

# 研究智能体
research_agent = create_react_agent(
  model,
  tools=[web_search],
  prompt=(
      "You are a research agent.\n\n"
      "INSTRUCTIONS:\n"
      "- Assist ONLY with research-related tasks, DO NOT do any math\n"
      "- After you're done with your tasks, respond to the supervisor directly\n"
      "- Respond ONLY with the results of your work, do NOT include ANY other text."
  ),
  name="research_agent"
)

# for chunk in research_agent.stream(
#     {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
# ):
#     pretty_print_messages(chunk)

def add(a: float, b: float) :
  """" Adds two numbers """
  return a + b

def multiply(a: float, b:float) :
  """" Multiplies two numbers """
  return a * b

def divide(a: float, b: float) :
  """" Divides two numbers """
  return a / b

math_agent = create_react_agent(
  model,
  tools=[add, multiply, divide],
  prompt=(
    "You are a math agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with math-related tasks\n"
    "- After you're done with your tasks, respond to the supervisor directly\n"
    "- Respond ONLY with the results of your work, do NOT include ANY other text."
  ),
  name="math_agent"
)

# for chunk in math_agent.stream(
#     {"messages": [{"role": "user", "content": "what's (3 + 5) x 7"}]}
# ):
#     pretty_print_messages(chunk)


supervisor = create_supervisor(
   model=model,
   agents=[research_agent, math_agent],
   prompt=(
     "You are a supervisor managing two agents:\n"
      "- a research agent. Assign research-related tasks to this agent\n"
      "- a math agent. Assign math-related tasks to this agent\n"
      "Assign work to one agent at a time, do not call agents in parallel.\n"
      "Do not do any work yourself."
   ),
   add_handoff_back_messages=True,
   output_mode="full_history"
).compile()

graph_png = supervisor.get_graph().draw_mermaid_png()
with open("multi_agent.png", "wb") as f:
  f.write(graph_png)

for chunk in supervisor.stream(
  {
    "messages": [
      {
        "role": "user",
        "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
      }
    ]
  }
):
  pretty_print_messages(chunk, last_message=True)
