from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
import os
from langgraph.prebuilt import create_react_agent
import operator
from typing import Annotated, List, Dict, Tuple, TypedDict
import asyncio

model=ChatOpenAI(
  api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  model="qwen-plus", # 替换为所需模型名称
  temperature=0
)

web_search = TavilySearch(max_results=1)
prompt = "You are a helpful assistant."

agent_executor = create_react_agent(
  model=model,
  tools=[web_search],
  prompt=prompt
)

#定义一个TypedDict，用于存储输入，计划，过去的步骤和响应
class PlanExecution(TypedDict):
  input: str
  plan: List[str]
  past_steps: Annotated[List[Tuple], operator.add]
  response: str

from pydantic import BaseModel, Field

#定义一个BaseModel，用于存储计划
class Plan(BaseModel):
  """ 未来要执行的计划"""

  steps: List[str] = Field(
    description="需要执行的不同步骤，应该按顺序排列"
  )

from langchain_core.prompts import ChatPromptTemplate
#定义一个ChatPromptTemplate，用于生成输入
planner_prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
       """对于给定的目标，提出一个简单的额逐步计划，这个计划应该包含独立的任务，如果正确执行将得到正确的答案，不要添加多余的步骤。
       最后一步的结果应该是最终答案。确保每一步都有所必要的信息——不要跳过步骤。
       请确保返回的JSON对象中包含一个名为“steps”的键，其值为字符串数组。
      """ #新增提示 
    ),
    ("placeholder", "{messages}"),
  ]
)

#使用指定的提示模版创建一个计划生成器
planner = planner_prompt | model.with_structured_output(Plan)

# planner.invoke({"messages": [("user", "现任澳网冠军的家乡在哪里?")]})

from typing import Union

#定义一个响应模型类，用于描述用户的响应
class Response(BaseModel):
  """ 用户的响应"""

  response: str


#定义一个行为模型类，用于描述用户的行为。
#该类中有一个action， 类型为Unior[Response, Plan], 表示可是是Response或者Plan类
# action 熟悉的描述为: 要执行的行为，如果要回应用户，用Response类，如果需要进一步使用工具获取答案，使用Plan类。
class Act(BaseModel):
  """ 用户的行为"""

  action: Union[Response, Plan] = Field(
    description="要执行的行为，如果要回应用户，用Response类，如果需要进一步使用工具获取答案，使用Plan类。"
  )

  #创建一个重新计划的提示模版
replanner_prompt = ChatPromptTemplate.from_messages([
   (
      "system",
      """对于给定的目标，提出一个简单的逐步计划.这计划应该包含独立的任务,如果正确执行将得到正确的答案。不要添加任务多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所必要的信息——不要跳过步骤。
      你必须以JSON格式输出，且JSON必须包含一个`action`字段，该字段可以有两种类型：`response`和`plan`。
       1. 如果任务已经完成，不需要更多步骤，则使用`Response`类型，例如：{{"action": {{"response": "最终答案"}}}}
       2. 如果还需要进一步步骤，则使用`Plan`类型，例如：{{"action": {{"steps": ["步骤1", "步骤2"]}}}}
      你的目标是:
      {input}

      你的原计划是:
      {plan}
      你的过去的步骤是:
      {past_steps}

      相应的更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加添加需要完成的步骤。
      不要返回已完成的步骤。
      """
    )
  ])

replanner = replanner_prompt | model.with_structured_output(Act)

from typing import Literal


async def main():
   #定义一个异步函数，用于生成计划步骤
    async def plan_step(state: PlanExecution):
      plan = await planner.ainvoke({"messages": [("user", state["input"])]})
      print("plan", plan)
      return {"plan": plan.steps}
    
    #定义一个异步函数，用于执行计划步骤
    async def execute_step(state: PlanExecution):
      plan = state["plan"]
      plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
      task = plan[0]
      task_formatted = f"""对于一下计划:
       {plan_str}\n\n
        你的任务是执行第{1}步，{task}。"""
      agent_response = await agent_executor.ainvoke(
         {"messages": [("user", task_formatted)]}
      )
      return {
        "past_steps": state["past_steps"] + [(task, agent_response["messages"][-1].content)],
      }
   
    #定义一个异步函数，用于重新计划步骤
    async def replan_step(state: PlanExecution):
       output = await replanner.ainvoke(state)
       if isinstance(output.action, Response):
          return {"response": output.action.response}
       else:
          return {"plan": output.action.steps}
       
    #定义一个函数，用于判断是否结束
    def should_end(state: PlanExecution) -> Literal["agent", "__end__"]:
      if "response" in state and state["response"]:
          return "__end__"
      else:
          return "agent"

    from langgraph.graph import StateGraph, START

    workflow = StateGraph(PlanExecution)
    workflow.add_node("planner", plan_step)

    workflow.add_node("agent", execute_step)

    workflow.add_node("replanner", replan_step)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")

    workflow.add_edge("agent", "replanner")

    workflow.add_conditional_edges(
       "replanner",
       should_end
    )

    app = workflow.compile()
    graph_png = app.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
      f.write(graph_png)
    
    config = {"recursion_limit": 50}
    inputs = {"input": "2020年东京奥运会100米田径决决冠军的家乡是哪里？请用中文回复"}
    async for event in app.astream(inputs, config=config):
        for k , v in event.items():
           if k != "__end__":
             print(v)

asyncio.run(main())
    

    
   
