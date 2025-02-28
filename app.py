from openai import OpenAI
import streamlit as st
from multi_agent import debate_chatbot
from langchain_core.messages import AIMessage, HumanMessage
from prompts import stage0, stage1, stage2
import time

st.set_page_config(page_title = "AI 토론 튜터")

title = "AI 토론 튜터와 토론 연습을 해보자"

with st.container(border=True):
    st.markdown(
        f"""<h2 style='text-align: center; color: black; font-size: 1.7rem; fontpropertise: prop'>{title}</h2>""", unsafe_allow_html=True)
        
    st.image('img.PNG')


# ----------------------------------------------------------------------------------------------------
# Session State
# ----------------------------------------------------------------------------------------------------

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="안녕! 나는 오늘 너와 토론을 진행할 토론 파트너야. 만나서 반가워.")
    ]

# 메시지 히스토리 표시
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)


def convert_messages(messages):
    converted = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            converted.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            converted.append({"role": "user", "content": msg.content})
    return converted


# ----------------------------------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------------------------------

if prompt := st.chat_input():
    
    # #debatePrompt 확인
    # print("prompt:", prompt)
    # print("debatePrompt:", st.session_state.get("debatePrompt", stage0))
    
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):      
        
        #messages에서 content 내용만 추출하기 
        extracted_contents = convert_messages(st.session_state.messages)
        
        # 초기 상태 설정
        initial_state = {
            "messages": extracted_contents,
            "debatePrompt": st.session_state.get("debatePrompt", stage0),
            "debateTopic": "알고리즘의 추천이 우리의 삶을 풍요롭게 해줄까?",
            "webSearch": "",
            "questionWeb": "",
            "generatedDebate": ""
        }

        try:
            # 그래프 실행 및 상태 업데이트
            for step in debate_chatbot.stream(
                initial_state,
                config={
                    "recursion_limit": 100
                }
            ):

                for node_name, state in step.items():  
                    #webgenerate가 출력을 잘하고 있는지 파악하기
                    if  node_name in ["shouldiwebsearch"]:
                        print("prompt:", prompt)
                        print("state[webSearch]", state)

                    if node_name in ["web_generate", "self_generate"]:
                        last_msg = state['generatedDebate']

                        # Generator를 활용한 Streaming 출력
                        def stream_generated_debate():
                            for word in last_msg.split():
                                yield word + " "  # 단어 단위 출력
                                time.sleep(0.1)  # 자연스러운 흐름

                        stream = stream_generated_debate()
                        response = st.write_stream(stream)  # Generator를 Streamlit에 전달

                        # AIMessage 추가
                        st.session_state.messages.append(AIMessage(content=response))

                        if "본격적인 토론을 시작할게" in response:
                            st.session_state.debatePrompt = stage1
                        if "반론 및 재반론 연습을 시작해보자" in response:
                            st.session_state.debatePrompt = stage2
                
                        
                

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")