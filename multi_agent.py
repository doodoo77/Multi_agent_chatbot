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

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    debateTopic: str
    debatePrompt: str
    webSearch: str
    questionWeb: str
    generatedDebate: str
    # Agentic RAG
    vectorDB: Annotated[Sequence[BaseMessage], add_messages]  # (토론 연습) db 사용함, (절차 안내) db 사용안함
    retrieve: str  # (관련 있음) 유효성 평가와 관련된 db임, (관련 없음)
    documents: List[str]  # (벡터DB에서 추출된 문서 리스트)
    valid: str  # (Pass) 생성된 토론이 유효성 검증을 통과함, (Fail)

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
          : 또한 어떤 상황인지 상관없이 무조건 대사를 생성해야해.

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
          : 또한 어떤 상황인지 상관없이 무조건 대사를 생성해야해.

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


#Agentic RAG 구축
# vertorDB 를 위한 전처리/ 벡터DB에 저장하기

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

PDFs = [
    "C:\Multi_agent_LLMs\DATA_vertorDB\classroom_debate_rubric.pdf",
    "C:\Multi_agent_LLMs\DATA_vertorDB\information-14-00503.pdf",
]


docs = [PyMuPDFLoader(PDF).load() for PDF in PDFs]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="db"  # ← 이거 추가하면 저장 위치 지정됨
)

vectorstore.persist()

retriever = vectorstore.as_retriever()

#Retriever를 도구로 저장하기
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name = "guideline_pdf",
    description = """챗봇 응답의 교육적 유효성을 평가할 때 해당 tool을 사용해 \n
                  특히, 챗봇의 응답이 토론 절차를 안내하는 사회자 역할을 하는게 아니라, \n
                  학생에게 피드백을 제공하거나 반론 및 재반론을 제시하는 상황일 때, \n
                  해당 tool을 사용해서 챗봇 응답의 유효성을 평가해""",
)

tools = [retriever_tool]

#Agent: 유효성 평가를 해야하는지 말아야 하는지를 평가함 
def agent(state: State):
  """
  생성된 토론 챗봇의 응답에 대해 유효성을 평가해야하는 평가하고, 관련 평가 기준을 불러옴

  Args:
    state (messages): The current state

  Returns:
    dict: The documents for assessing chatbot's validation
  """
  global llm
  eval_prompt = PromptTemplate(
    template = """
   너는 챗봇이 진행하는 토론의 유효성을 평가하는 역할을 맡고 있어. \n
    먼저, 현재 상황이 토론의 유효성을 평가해야 하는 경우인지 판단해야 해. \n

    이를 위해 [전체 토론 기록]과 현재 시점에서의 [챗봇의 응답]을 참고하여, \n
    챗봇이 단순히 사회자처럼 토론 절차를 안내하는 중인지, \n
    아니면 직접 토론을 진행하며 학생과 의견을 주고받는 중인지 파악해. \n

    만약 챗봇이 단순히 토론 절차를 안내하는 역할이라면 \n
    유효성 평가를 진행할 필요 없이 그대로 넘어가면 돼. \n
    예를 들어, 챗봇이 생성한 토론이 \n
    "안녕하세요". \n
    "읽기 자료를 읽어보세요". \n
    "우리의 주장을 지지하는 근거를 하나만 말해보겠어?". \n
    "혹시 해당 근거가 우리의 주장과 어떻게 연결되는지 구체적으로 설명해줄래?" \n
    "근거를 지지할 실제 사례를 추가해줄래?". \n
    "혹시 해당 사례가 우리의 근거와 어떻게 연결되는지 구체적으로 설명해줄래?" \n
    "고생했어. 우리 이제 반론 및 재반론 연습을 시작해보자." \n
    와 같은 상황일때가 단순히 토론 절차를 안내하는 상황이야. \n
    위와 같은 상황에서 너의 출력을 20자 이내로 한정해. \n  

    하지만 챗봇이 학생에게 피드백을 제공하거나 반론 및 재반론하는 상황이라면, \n
    그 응답이 교육적으로 적절한지 검토해야 해. 이를 위해 먼저 전체 토론 기록을 살펴보고 \n
    현재 토론이 어떤 단계에 있는지 파악해. 그런 다음 해당 응답이 교육적으로 적절한지, \n
    학생의 의도와 맥락에 적합한 응답인지, 논리적으로 타당한지, 윤리적으로 타당한 응답인지 등을 판단하면 돼. \n
    위와 같은 상황에서의 출력 제한은 없어. \n
    또한, 이 상황에 해당하면 반드시 vectorDB에 저장된 정보들을 불러와야 해. 즉, tool_calls를 반드시 호출해

    [전체 토론 기록]
    : {history_debate}

    [챗봇의 응답]
    : {generatedDebate}
    """,
    input_variables=["history_debate", "generatedDebate"],
    )

  formatted_prompt = eval_prompt.format(history_debate=state["messages"],
                                        generatedDebate=state["generatedDebate"])

  llm = llm.bind_tools([retriever_tool])
  agent = llm.invoke(formatted_prompt)

  return {
      "vectorDB": agent,
      "generatedDebate": state["generatedDebate"]
      }

#Retrieve: 불러온 정보가 유효성 평가를 위한 것인지 판단함 

from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

def retrieve_node(state: State):
  """
  벡터db에서 추출된 정보가 챗봇 응답의 유효성을 평가하기에 적절한지 평가함

  Args:
    state (messages): The current state

  Returns:
    str: '관련 있음' or '관련 없음'
  """

  class grade(BaseModel):
    binary_score: str = Field(description="""불러온 문서가 챗봇 응답의 유효성을 평가하는것과 \n
                                          관련 있으면 '관련 있음'을 출력해 \n
                                          관련 없으면 '관련 없음'을 출력해""")
  llm_with_tool = llm.with_structured_output(grade)

  prompt = PromptTemplate(
      template = """
      당신은 [불러온 문서]가 [챗봇이 생성한 토론]의 유효성을 평가하는 데 적절한지를 판단하는 평가자입니다.

      즉, 현재 불러온 문서가 챗봇이 [현재 토론 맥락]에서 학생들에게 교육적으로 도움이 되는 토론을 생성했는지 평가하기 위한 정보를 포함하고 있는지 파악하면 됩니다.
      이 작업은 엄격한 테스트가 아니라, 챗봇의 토론을 적절히 평가할 수 없는 문서를 걸러내는 것이 목표입니다.

      제시된 문서가 챗봇의 토론 유효성을 평가하는 기준과 직접적으로 관련이 있거나 도움이 된다면 '관련 있음'으로 평가하세요.
      반대로, 문서가 챗봇의 토론을 평가하는 데 적절하지 않거나 관련성이 부족하다면 '관련 없음'으로 평가하세요.
      평가는 '관련 있음' 또는 '관련 없음'의 이진 점수로만 표시하세요.

      [불러온 문서]
      {document}

      [챗봇이 생성한 토론]
      {generatedDebate}

      [현재 토론 맥락]
      {history_debate}
      """,
      input_variables=["document", "generatedDebate", "history_debate"],
  )
  #Chain
  chain = prompt|llm_with_tool

  retrieve = ToolNode([retriever_tool])

  document = retrieve.invoke({"messages": state['vectorDB']})

  for message in document['messages']:
    content = message.content

  generatedDebate = state["generatedDebate"]

  scored_result = chain.invoke({"document": content, "generatedDebate": generatedDebate, "history_debate": state["messages"]})

  score = scored_result.binary_score

  return {
      "retrieve": score,
      "documents": content
  }

#Generate: 유효성 평가를 진행함 
def generate(state: State):
  """
  불려진 가이드라인 정보를 기반으로 챗봇이 생성한 토론의 유효성을 평가함

  Args:
    state: The current state

  Returns:
    valid: pass or fail
  """
  class grade(BaseModel):
    binary_score: str = Field(description="""챗봇 응답의 교육적 이점이 있는 유효성을 평가해. \n
                                          유효하면 'pass'을 출력해 \n
                                          유효하지 않으면 'fail'을 출력해""")

  llm_with_tool = llm.with_structured_output(grade)

  eval_prompt = PromptTemplate(
    template = """
    너는 [평가 가이드라인]를 참고해서 챗봇이 진행하는 토론의 유효성을 평가하는 역할을 맡고 있어. \n

    이를 위해 [전체 토론 기록]과 현재 시점에서의 [챗봇의 응답]을 참고하여, \n
    챗봇이 토론 맥락에 적절하고 해당 응답이 교육적으로 적절한지, \n
    학생의 의도와 맥락에 적합한 응답인지, 논리적으로 타당한지, \n
    윤리적으로 타당한 응답인지 등을 주어진 [평가 가이드라인]을 참고해서 이진 평가하면 돼.

    챗봇의 응답이 교육적으로 유효하다면 "pass",
    유효하지 않다면 "fail"로 답변해.

    [평가 가이드라인]
    : {retrieve_docs}

    [전체 토론 기록]
    : {history_debate}

    [챗봇의 응답]
    : {generatedDebate}
    """,
    input_variables=["retrieve_docs", "history_debate", "generatedDebate"],
    )

  chain = eval_prompt|llm_with_tool

  input_data = {
      "retrieve_docs": state["documents"],
      "history_debate": state["messages"],
      "generatedDebate": state["generatedDebate"]
  }

  response = chain.invoke(input_data)

  return {
      "valid" : response.binary_score
  }

def generate_edge(state: State):
  binary_score = state["valid"]

  if binary_score == "pass":
    return "END" 
  else:
    if state["wedSearch"] == "yes":
      return "web_generate"
    else:
      return "self_generate" #자체적으로 생성하는 상황


def retrieve_edge(state: State):
  score = state["retrieve"]

  if score == "관련 있음":
      return "generate"

  else:
      print(score)
      return "agent"


#그래프 구축
binary_toolcall = " "
def route_tools(state: State):
    global binary_toolcall
    ai_message = state['vectorDB']
    for message in ai_message:
            if hasattr(message, 'tool_calls') and isinstance(message.tool_calls, list) and len(message.tool_calls) > 0:
                print("route_tools의 함수값은 tools입니다.")
                binary_toolcall = "tools"
                return "tools"

    print("route_tools의 함수값은 END입니다.")
    binary_toolcall = END 
    return END

def reset_state():
    global binary_toolcall
    binary_toolcall = " "  # 초기값으로 재설정

# Create the graph
from langgraph.graph import StateGraph, START, END
#from langgraph.prebuilt import tools_condition

# Initialize graph
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("shouldiwebsearch", shouldiwebsearch)
graph_builder.add_node("web_generate", web_generate)
graph_builder.add_node("self_generate", self_generate)

# Add all agentic nodes
graph_builder.add_node("agent", agent)
graph_builder.add_node("retrieve_node", retrieve_node)
graph_builder.add_node("generate", generate)


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

#Add edges (self_generate & web_generate to agent)
graph_builder.add_edge("web_generate", "agent")
graph_builder.add_edge("self_generate", "agent")

# Add conditional edges based on DB 사용여부
graph_builder.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "retrieve_node",
        END: END,
    },
)

# Add conditional edges based on 관련 여부
graph_builder.add_conditional_edges(
    "retrieve_node",
    retrieve_edge,
    {
        "generate": "generate",
        "agent" : "agent"
    }
)

# Add conditional edges based on 관련 여부
graph_builder.add_conditional_edges(
    "generate",
    generate_edge,
    {
        "END": END,
        "web_generate": "web_generate",
        "self_generate": "self_generate"
    }
)


# Add edges to END
# graph_builder.add_edge("web_generate", END)
# graph_builder.add_edge("self_generate", END)

# Compile the graph
debate_chatbot = graph_builder.compile()