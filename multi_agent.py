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

from prompts import stage0, stage1, stage2

import warnings
warnings.filterwarnings('ignore')

# State definition
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    debateTopic: str
    debatePrompt: str
    webSearch: str
    questionWeb: str
    generatedDebate: str

llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets['OPENAI_API_KEY'], 
    organization=st.secrets['OPENAI_ORGANIZATION'])

import os
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']

def shouldiwebsearch(state: State):
  """
  웹 검색 여부를 파악해서 Yes or No
  Yes: 웹검색이 필요함
  No: 자체적으로 생성 가능함
  """

  #outparser정의하기
  class answer_availability(BaseModel):
    binary_answer: str = Field(description="""
                              웹검색을 기반으로 학생에게 답변을 제시해야 한다면 'Yes'를 출력하고,
                              웹검색 없이 자체적으로 답변을 생성할 수 있으면 'No'를 출력해
                              """)

    question_web: str = Field(description="""
                              웹 검색이 필요할 경우, 학생의 응답에 적절하게 대응하기 위해,
                              웹에 어떤 정보를 검색할지 100자 이하로 생성해봐
                              """)

  web_parser = JsonOutputParser(pydantic_object=answer_availability)


  evaluator = llm.with_structured_output(answer_availability)
  eval_prompt = PromptTemplate(
      template="""
      너는 챗봇이 토론 연습을 제공할 때, 웹검색을 기반으로 답변을 생성해야하는지 여부를 파악하는 평가자야.

      웹 검색이 필요한지는 두 가지 요소로 결정돼.
      1. 토론 절차: {full_debate_process}를 참고했을 때, 챗봇이 학생에게 피드백을 제공하거나 반론 및 재반론을 제시해야 하는 상황
      2. 챗봇이 자체적으로 신뢰할 만한 실제 사례를 기반으로 답변을 생성할 수 없고, 웹 검색을 통해 내용을 보강해야 하는 상황
      즉, 이 두 가지 상황에 해당하면 웹 검색이 필요한 상황이야.

      웹 검색이 필요할 경우, {topic_debate}를 고려하여
      학생의 응답에 대응할 실제 사례를 찾기 위한 검색 질문을 생성해.
      웹 검색이 필요없으면 질문을 비워둬.

      답변을 생성할 때는 아래 지침을 따라.
      {format_instructions}

      너가 판단해야 하는 토론 상황은 아래와 같아.
      {history_debates}
      """,
      input_variables=["full_debate_process", "topic_debate", "history_debates"], #responce는 학생의 마지막 응답 이어야하겠네?
      partial_variables={"format_instructions": web_parser.get_format_instructions()},
  )
  full_debate_process = stage0 + stage1 + stage2

  #아래와 같은 형식으로 작성하는게 맞나?
  evaluation = evaluator.invoke(
  eval_prompt.format(full_debate_process = full_debate_process, topic_debate = state["debateTopic"], history_debates=state["messages"]) #이게 사용자의 마지막 응답이려나?
  )

  # print("webSearch:", evaluation.binary_answer)
  # print("questionWeb:", evaluation.question_web)

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


def web_generate(state: State):

  search_tool = TavilySearchResults(max_results=3)
  search_results = search_tool.invoke(state['questionWeb'])

  prompt_template = PromptTemplate(
        template="""
        [임무소개]
          : 너는 특정 [토론주제]에 대해서, [웹검색 결과]를 기반으로 학생에게 토론연습을 제공하는 토론 튜터야.
          : 너는 현재 [전체 토론 절차] 중 [특정 토론 절차]를 학생에게 제공해야해.
          : [특정 토론 절차]에는 너가 학생의 응답에 어떻게 반응해야 하는지가 명시되어 있어. 
          : [특정 토론 절차]를 참고하여 현재 [토론 기록]이 해당 절차의 어느 단계에 속하는지 판단한 후, 그에 맞는 다음 대사를 제공해.
          : 주의할 점은. 너가 대사를 한번에 내뱉는게 아니라 학생의 답변이 올 때까지 기다린 후에 다음 대사를 해야해.

        [토론주제]
          : 너가 학생과 토론하게 될 토론 주제는 {debateTopic}야.
      
        [웹검색 결과]
        웹 검색 결과는 학생에게 토론을 제공할 때, 실제 사례와 구체적인 사실을 기반으로 논의를 전개하기 위한 목적이야. 따라서, 웹 검색 결과를 활용할 때는 이 목적을 고려하여 신뢰할 수 있는 정보와 사례를 중심으로 토론 연습을 제공해.
        웹 검색 결과: {web_results}

        [전체 토론 절차]
          : 먼저 전체적인 토론절차에 대해서 소개해줄게.
          0. Reading material

          1. Constructive debate
          1.1. 이 단계에서는 너는 학생들의 주장에 해당하는 근거를 함께 세울거야.
          1.2. 그리고 나서 해당 근거를 지지하는 실제 사례를 추가할거야.

          2. Rebuttal debate
          2.1. 이제 너는 학생의 입장(반대 혹은 찬성)과 다른 입장에서 근거를 제시하고, 학생이 너의 근거에 대해서 반론을 제기하면, 너는 재반론을 해.
          2.2. 반대로 너가 학생의 근거에 대해 반론을 제기하고, 학생이 이에 대해서 재반론하는 단계를 가져.

        [특정 토론 절차]
          : 이제 너가 학생에게 제공해야하는 토론 절차는 아래와 같아
          주의! 내가 해준 대사를 변경하지 말고 그대로 읽어야 해. 큰 따옴표는 빼고 읽어.
          {debate_stage}

        [토론 기록]
          : {debate_history}  
        """,
        input_variables=["debate_history", "web_results", "debateTopic", "debate_stage"],
    )

  web_generate = llm.invoke(prompt_template.format(debate_history = state["messages"],
                                                   web_results = search_results,
                                                   debateTopic = state["debateTopic"],
                                                   debate_stage = state["debatePrompt"]))

  return {
      "generatedDebate": web_generate.content
  }


#토론 연습 without 웹검색 

def self_generate(state: State):

  prompt_template = PromptTemplate(
        template="""
        [임무소개]
          : 너는 특정 [토론주제]에 대해서 학생에게 토론연습을 제공하는 토론 튜터야.
          : 너는 현재 [전체 토론 절차] 중 [특정 토론 절차]를 학생에게 제공해야해.
          : [특정 토론 절차]에는 너가 학생의 응답에 어떻게 반응해야 하는지가 명시되어 있어. 
          : [특정 토론 절차]를 참고하여 현재 [토론 기록]이 해당 절차의 어느 단계에 속하는지 판단한 후, 그에 맞는 다음 대사를 제공해.
          : 주의할 점은. 너가 대사를 한번에 내뱉는게 아니라 학생의 답변이 올 때까지 기다린 후에 다음 대사를 해야해.

        [토론주제]
          : 너가 학생과 토론하게 될 토론 주제는 {debateTopic}야.

        [전체 토론 절차]
          : 먼저 전체적인 토론절차에 대해서 소개해줄게.
          0. Reading material

          1. Constructive debate
          1.1. 이 단계에서는 너는 학생들의 주장에 해당하는 근거를 함께 세울거야.
          1.2. 그리고 나서 해당 근거를 지지하는 실제 사례를 추가할거야.

          2. Rebuttal debate
          2.1. 이제 너는 학생의 입장(반대 혹은 찬성)과 다른 입장에서 근거를 제시하고, 학생이 너의 근거에 대해서 반론을 제기하면, 너는 재반론을 해.
          2.2. 반대로 너가 학생의 근거에 대해 반론을 제기하고, 학생이 이에 대해서 재반론하는 단계를 가져.

        [특정 토론 절차]
          : 이제 너가 학생에게 제공해야하는 토론 절차는 아래와 같아
          주의! 내가 해준 대사를 변경하지 말고 그대로 읽어야 해. 큰 따옴표는 빼고 읽어.
          {debate_stage}

        [토론 기록]
          : {debate_history}  
        """,
        input_variables=["debate_history", "debateTopic", "debate_stage"],
    )

  self_generate = llm.invoke(prompt_template.format(debate_history = state["messages"],
                                                   debateTopic = state["debateTopic"],
                                                   debate_stage = state["debatePrompt"]))


  return {
      "generatedDebate": self_generate.content
  }

#그래프 구축

# Create the graph
from langgraph.graph import StateGraph, START, END

# Initialize graph
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("shouldiwebsearch", shouldiwebsearch)
graph_builder.add_node("web_generate", web_generate)
graph_builder.add_node("self_generate", self_generate)

# Add edges
# Start flow
graph_builder.add_edge(START, "shouldiwebsearch")

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