from openai import OpenAI
import streamlit as st
from multi_agent import debate_chatbot
from langchain_core.messages import AIMessage, HumanMessage

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


# ----------------------------------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------------------------------

if prompt := st.chat_input():

    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):      
        # 초기 상태 설정
        initial_state = {
            "messages": prompt,
            "debateTopic": "알고리즘의 추천이 우리의 삶을 풍요롭게 해줄까?",
            "debateProcess": "",
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
                    if node_name == "indentify_debateprocess":
                        st.write(state['debateProcess'])
                    
                    if node_name == "shouldiwebsearch":
                        st.write(state['webSearch'])

                    if node_name == "web_generate" or node_name == "self_generate":
                        last_msg = state['generatedDebate']
                        st.session_state.messages.append(AIMessage(content=last_msg))
                        st.markdown(last_msg)        

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")