from typing import Annotated, TypedDict, List, Dict, Sequence
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages 
from langchain_core.messages import BaseMessage 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI 

from pydantic import BaseModel, Field 

import streamlit as st 

import warnings
warnings.filterwarnings('ignore')

# State definition
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    debateTopic: str
    debateProcess: str
    webSearch: str
    questionWeb: str
    generatedDebate: str
    verifyValidity: str

llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets['OPENAI_API_KEY'], 
    organization=st.secrets['OPENAI_ORGANIZATION'])

import os
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']


#토론 절차 인식
class create_DP(BaseModel):
  process: str = Field(description= """챗봇과 학생의 토론 내용을 분석하여 현재 토론 절차를 식별해.
                                    절차는 '근거 제시', '사례 제시', '반론 제시', '재반론 제시' 중 하나로 분류하고,
                                    각 절차가 몇 번째인지도 함께 표시해. (예: '근거 제시 첫번째')
                                    그리고 가장 최신의 토론 단계만 말해주면 돼. 즉, 하나만 말해야 한다는 거지.
                                    """)

def indentify_debateprocess(state: State):
  # DynamicProcess = create_DP(state["messages"])
  process_parser = JsonOutputParser(pydantic_object=create_DP)

  # 토론 내용을 기반으로 토론 절차를 구분하기 위한 prompt
  # 근거 제시 단계인지 ...
  process_prompt = PromptTemplate(
      template="""
      너는 토론 내용을 분석하여 현재 토론 절차를 구분하면 돼.
      
      토론 절차는 다음과 같아. 즉, 한번 넘어간 토론 절차는 다시 뒤로 넘어가면 안돼.
      토론 절차: 
        1. 근거 제시 첫번째
        2. 사례 제시 첫번째
        3. 근거 제시 두번째
        4. 사례 제시 두번째
        5. 반론 제시 첫번째
        6. 반론 제시 두번째
        7. 재반론 제시 첫번째
        8. 재반론 제시 두번째

      각 절차의 정의는 다음과 같아.

      근거 제시: 챗봇이 학생에게 근거를 요청하면, 학생이 이를 제시하는 단계야.
      근거가 부족할 경우, 챗봇이 구체적인 설명을 요구할 수 있고,
      학생이 추가 보완하거나 질문을 제시할 수도 있어.
      챗봇이 다음 근거를 요청하기 전까지를 하나의 근거 제시 단계로 간주해.

      사례 제시: 학생이 제시한 근거를 보완하기 위해 실제 사례를 추가하는 단계야.
      사례가 부족하면 챗봇이 구체적인 예시를 요구할 수 있고,
      학생이 추가 보완하거나 질문을 제시할 수도 있어.
      챗봇이 다음 사례를 요청하기 전까지를 하나의 사례 제시 단계로 간주해.

      반론 제시: 학생이 챗봇의 근거에 반론을 제기하는 단계야. 같은 주제 내에서 여러 차례 반론이 오갈 수 있으며, 주제가 바뀔 때까지 하나의 반론 제시 단계로 간주해.
      재반론 제시: 챗봇이 학생의 근거에 반론을 제기하면, 학생이 이에 대해 재반론하는 단계야. 이 또한 주제가 바뀔 때까지 하나의 재반론 제시 단계로 간주해.

      이제 아래의 토론 내용의 토론 절차를 식별해 줘.
      {debate_histoty}

      아래 형식에 맞춰서 토론 절차를 인식해.
      {format_instructions}
      """,
      input_variables=["debate_histoty"],
      partial_variables={"format_instructions": process_parser.get_format_instructions()},
  )

  chain = process_prompt | llm | process_parser

  debateProcess = chain.invoke({
      "debate_histoty": state["messages"] #여기에 들어가는건 챗봇과 학생의 토론 내용인데,
                                          #내가 원하는 대로 들어가고 있는지 파악해야 함

  })
  return {"debateProcess": debateProcess["process"]}

#웹 검색 여부 파악
def shouldiwebsearch(state: State):
  """
  웹 검색 여부를 파악해서 Yes or No
  Yes: 웹검색이 필요함
  No: 자체적으로 생성 가능함
  """

  #outparser정의하기
  class answer_availability(BaseModel):
    binary_answer: str = Field(description="""
                              자체적으로 구체적인 사례를 기반으로 학생의 응답에 대응할 수 있으면 'No'를 출력해.
                              그렇지 않고 웹검색이 필요하면 'Yes'를 출력해.
                              """)
    question_web: str = Field(description="""
                              웹 검색이 필요할 경우, 학생의 응답에 적절하게 대응하기 위해,
                              웹에 어떤 정보를 검색할지 100자 이하로 생성해봐
                              """)

  web_parser = JsonOutputParser(pydantic_object=answer_availability)


  evaluator = llm.with_structured_output(answer_availability)
  eval_prompt = PromptTemplate(
      template="""
      당신이 학생의 응답에 대해서 아래와 같은 작업을 한다고 했을때,
      자체적으로 신뢰할 수 있는 실제 사례를 기반으로 답변을 생성할 수 있는지,
      아니면 웹 검색을 통해 내용을 보강해야 하는지 평가하는 역할을 합니다.

      학생이 제시한 근거에 대한 피드백 제공
      학생이 제시한 사례 보충
      학생의 근거에 대한 반론 제기
      학생의 반론에 대한 재반론

      웹 검색이 필요할 경우, {topic_debate}를 고려하여
      학생의 응답에 대응할 실제 사례를 찾기 위한 검색 질문을 생성해.
      웹 검색이 필요없으면 질문을 비워둬.

      답변을 생성할 때는 아래 지침을 따라.
      {format_instructions}

      또, 학생의 응답은 아래와 같아.
      {responce}
      """,
      input_variables=["topic_debate", "responce"], #responce는 학생의 마지막 응답 이어야하겠네?
      partial_variables={"format_instructions": web_parser.get_format_instructions()},
  )

  evaluation = evaluator.invoke(
  eval_prompt.format(topic_debate = state["debateTopic"], responce=state["messages"][-1]) #이게 사용자의 마지막 응답이려나?
  )

  return {
      "webSearch": evaluation.binary_answer,
      "questionWeb": evaluation.question_web
  }

#웹 검색 여부를 기반으로 다음 노드 라우팅
def route_after_search(state: State):

  if state["webSearch"] == "Yes":
    return "web_generate" #웹 검색이 필요한 상황
  else:
    return "self_generate" #자체적으로 생성하는 상황

#토론 검색 with 웹검색
example_evidence = """
예시 1.
    <
    챗봇: 좋아. 먼저 우리의 주장을 지지하는 근거를 하나만 말해보겠어?
    학생: 대부분의 포털 사이트에서도 사용자가 접속한 페이지를 바탕으로 메인 화면에 맞춤형 콘텐츠를 안내하고 있어
    >
    위 대화에서 학생이 맞춤형 콘텐츠를 안내하는게 왜 우리의 삶에 악영향을 주는지 설명하지 않고 있어. 이때 너가 구체적으로 설명해달라고 말하면 돼.

예시 2.
    <
    챗봇: 좋아. 먼저 우리의 주장을 지지하는 근거를 하나만 말해보겠어?
    학생: 알고리즘은 우리를 우매하게 만듭니다. 생각할 필요가 없어지기 때문이죠.
    >
    위 대화에서 학생이 알고리즘의 추천이 우리가 생각하지 않게 만드는지 설명하고 있지 않아. 이때 너가 구체적으로 설명해달라고 말하면 돼.

예시 3.
    <
    챗봇: 좋아. 먼저 우리의 주장을 지지하는 근거를 하나만 말해보겠어?
    학생: 두번째 근거는, 편향이 생길 수 있다는 점이 있어
    >
    위 대화에서 학생이 알고리즘의 추천에 편향이 생기면 어떤 문제가 생기는지 설명하고 있지 않아. 이때 너가 구체적으로 설명해달라고 말하면 돼.
"""

example_warrant = """
예시 1.
    <
    챗봇: 우리의 주장을 더 구체적으로 지지하기 위해서 이 근거와 관련된 실제 사례를 하나 추가해줄래?
    학생: 대표적으로 유튜브나 인스타그램이 있지
    >
    위 대화에서 학생이 유튜브나 인스타그램이 학생의 근거를 어떻게 지지하는지 설명하지 않고 있어.

예시 2.
    <
    챗봇: 우리의 주장을 더 구체적으로 지지하기 위해서 이 근거와 관련된 실제 사례를 하나 추가해줄래?
    학생: 자극적인 콘텐츠가 생성돼
    >
    위 대화에서 학생은 알고리즘의 추천으로 인해 자극적인 콘텐츠가 왜 생성되는지 설명하고 있지 않고, 또 이게 왜 문제가 되는지 설명하지 않아.

예시 3.
    <
    챗봇: 우리의 주장을 더 구체적으로 지지하기 위해서 이 근거와 관련된 실제 사례를 하나 추가해줄래?
    학생: 구글과 같은 사례가 있지
    >
    위 대화에서 학생은 구글과 같은 사례가 본인의 근거와 어떻게 연결되는지 언급하고 있지 않아.
"""

def web_generate(state: State):

  search_tool = TavilySearchResults(max_results=3)
  search_results = search_tool.invoke(state['questionWeb'])

  prompt_template = PromptTemplate(
        template="""
        너는 웹검색 결과를 기반으로 학생에게 토론연습을 제공하는 토론 튜터야.
        너는 학생과 나눈 토론 기록을 보고 아래에 제시한 지시에 따라서 토론을 제공해주면 돼.
        토론 기록: {debate_history}

        웹 검색 결과는 학생에게 토론을 제공할 때, 실제 사례와 구체적인 사실을 기반으로 논의를 전개하기 위한 목적이야. 따라서, 웹 검색 결과를 활용할 때는 이 목적을 고려하여 신뢰할 수 있는 정보와 사례를 중심으로 토론 연습을 제공해.
        웹 검색 결과: {web_results}

        또한, 현재 토론 절차와 아래의 토론 순서를 고려해서 차례대로 토론 연습을 제공해.
        토론 절차가 아직 없으면, 근거 제시 첫번째부터 시작하면 돼.
        현재 토론 절차: {debate_process}
        토론 순서:
        1. 근거 제시 첫번째
        2. 사례 제시 첫번째
        3. 근거 제시 두번째
        4. 사례 제시 두번째
        5. 반론 제시 첫번째
        6. 반론 제시 두번째
        7. 재반론 제시 첫번째
        8. 재반론 제시 두번째

        이제 각 토론 단계에서 너가 어떤 역할을 수행해야 하는지 설명해줄게, 해당 역할들을 수행하면 다음 토론 단계로 넘어가서 토론 연습을 제공해주면 돼.

        근거 제시 단계는 학생에게 근거를 물어보고, 피드백을 제공하는 단계야.
        1. 학생에게 지지하는 근거를 물어봐: “먼저 우리의 주장을 지지하는 근거를 하나만 말해보겠어?”
        2. 학생의 응답이 구체적이지 않으면 피드백을 제공해: “혹시 해당 근거가 우리의 주장과 어떻게 연결되는지 구체적으로 설명해줄래?”
        피드백을 제공해야하는 예시를 제공해줄게 참고해.
        {example_evidence}

        사례 제시 단계는 학생에게 사례를 물어보고, 피드백을 제공하는 단계야.
        0. 근거 제시 단계에서 이미 사례가 제시됐으면, 이 단계를 넘어가
        1. 학생에게 근거를 지지하는 실제 사례를 물어봐: “우리의 근거를 지지할 실제 사례를 추가해줄래?”
        2. 학생이 제시한 사례가 근거와 직접적인 연결이 드러나지 않으면 피드백을 제공해: “해당 사례가 우리의 근거와 어떻게 연결되는지 구체적으로 설명해줄래?”
        피드백을 제공해야하는 예시를 제공해줄게 참고해.
        {example_warrant}

        반론 제시 단계에서는 학생이 너의 근거에 대해서 반론하는 단계야.
        1. 학생의 반대입장에 서서 근거를 하나 제시하고, 반론해보라고 학생에게 말해: “내 근거에 대해서 어떻게 생각하니? 반론해줄래?”
        2. 더 깊은 토론을 위해 학생이 제시한 반론에 대해서 2턴 이상 대응해

        재반론 제시 단계에서는 너가 학생의 근거에 대해 반론을 제기하고 학생이 이에 대해 재반론하는 단계야.
        1. 학생의 근거에 대해서 반론을 제기해.
        2. 더 깊은 토론을 위해 학생이 제시한 재반론에 대해서 2턴 이상 반론을 제기해

        그리고 각 토론 단계에 대해서 학생에게 안내할때는 핵심요지만 말하도록 해.""",
        input_variables=["debate_history", "web_results", "debate_process", "example_evidence", "example_warrant"],
    )

  web_generate = llm.invoke(prompt_template.format(debate_history = state["messages"],
                                                          web_results = search_results,
                                                          debate_process = state["debateProcess"],
                                                          example_evidence = example_evidence,
                                                          example_warrant = example_warrant))

  return {
      "generatedDebate": web_generate.content
  }

#토론 연습 without 웹검색 
def self_generate(state: State):

  prompt_template = PromptTemplate(
        template="""
        너는 학생에게 토론연습을 제공하는 토론 튜터야.
        너는 학생과 나눈 토론 기록을 보고 아래에 제시한 지시에 따라서 토론을 제공해주면 돼.
        토론 기록: {debate_history}

        또한, 현재 토론 절차와 아래의 토론 순서를 고려해서 차례대로 토론 연습을 제공해.
        토론 절차가 아직 없으면, 근거 제시 첫번째부터 시작하면 돼.
        현재 토론 절차: {debate_process}
        토론 순서:
        1. 근거 제시 첫번째
        2. 사례 제시 첫번째
        3. 근거 제시 두번째
        4. 사례 제시 두번째
        5. 반론 제시 첫번째
        6. 반론 제시 두번째
        7. 재반론 제시 첫번째
        8. 재반론 제시 두번째

        이제 각 토론 단계에서 너가 어떤 역할을 수행해야 하는지 설명해줄게, 해당 역할들을 수행하면 다음 토론 단계로 넘어가서 토론 연습을 제공해주면 돼.

        근거 제시 단계는 학생에게 근거를 물어보고, 피드백을 제공하는 단계야.
        1. 학생에게 지지하는 근거를 물어봐: “먼저 우리의 주장을 지지하는 근거를 하나만 말해보겠어?”
        2. 학생의 응답이 구체적이지 않으면 피드백을 제공해: “혹시 해당 근거가 우리의 주장과 어떻게 연결되는지 구체적으로 설명해줄래?”
        피드백을 제공해야하는 예시를 제공해줄게 참고해.
        {example_evidence}

        사례 제시 단계는 학생에게 사례를 물어보고, 피드백을 제공하는 단계야.
        0. 근거 제시 단계에서 이미 사례가 제시됐으면, 이 단계를 넘어가
        1. 학생에게 근거를 지지하는 실제 사례를 물어봐: “우리의 근거를 지지할 실제 사례를 추가해줄래?”
        2. 학생이 제시한 사례가 근거와 직접적인 연결이 드러나지 않으면 피드백을 제공해: “해당 사례가 우리의 근거와 어떻게 연결되는지 구체적으로 설명해줄래?”
        피드백을 제공해야하는 예시를 제공해줄게 참고해.
        {example_warrant}

        반론 제시 단계에서는 학생이 너의 근거에 대해서 반론하는 단계야.
        1. 학생의 반대입장에 서서 근거를 하나 제시하고, 반론해보라고 학생에게 말해: “내 근거에 대해서 어떻게 생각하니? 반론해줄래?”
        2. 더 깊은 토론을 위해 학생이 제시한 반론에 대해서 2턴 이상 대응해

        재반론 제시 단계에서는 너가 학생의 근거에 대해 반론을 제기하고 학생이 이에 대해 재반론하는 단계야.
        1. 학생의 근거에 대해서 반론을 제기해.
        2. 더 깊은 토론을 위해 학생이 제시한 재반론에 대해서 2턴 이상 반론을 제기해

        그리고 각 토론 단계에 대해서 학생에게 안내할때는 핵심요지만 말하도록 해.""",
        input_variables=["debate_history", "debate_process", "example_evidence", "example_warrant"],
    )

  self_generate = llm.invoke(prompt_template.format(debate_history = state["messages"],
                                                          debate_process = state["debateProcess"],
                                                          example_evidence = example_evidence,
                                                          example_warrant = example_warrant))

  return {
      "generatedDebate": self_generate.content
  }

#그래프 구축

# Initialize graph
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("indentify_debateprocess", indentify_debateprocess)
graph_builder.add_node("shouldiwebsearch", shouldiwebsearch)
graph_builder.add_node("web_generate", web_generate)
graph_builder.add_node("self_generate", self_generate)

# Add edges
# Start flow
graph_builder.add_edge(START, "indentify_debateprocess")
graph_builder.add_edge("indentify_debateprocess", "shouldiwebsearch")

# Add conditional edges based on certainty score
graph_builder.add_conditional_edges(
    "shouldiwebsearch",
    route_after_search,
    {
        "web_generate": "web_generate",
        "self_generate": "self_generate"
    }
)

# Add edges to END
graph_builder.add_edge("web_generate", END)
graph_builder.add_edge("self_generate", END)

# Compile the graph
debate_chatbot = graph_builder.compile()