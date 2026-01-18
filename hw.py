import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

st.title("English Learning Assistant")

api_key = st.secrets.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
)

user_input = st.chat_input("Напиши запит")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(
            """
            Ти є асистентом для вивчення англійської мови.
            Якщо користувач вводить окреме слово або коротку фразу, надай переклад
            та один приклад використання в англійському реченні.
            Якщо користувач вводить повне речення, надай переклад і коротке,
            зрозуміле пояснення використаної граматичної конструкції.
            Пояснюй просто, ніби людина бачить цю тему вперше.
            Відповідай структуровано і без зайвих відступів.
            """
        )
    ]

if user_input:
    st.session_state.chat_history.append(HumanMessage(user_input))
    ai_response = llm.invoke(st.session_state.chat_history)
    st.session_state.chat_history.append(ai_response)

for msg in st.session_state.chat_history:
    if isinstance(msg, SystemMessage):
        continue

    role = "human" if isinstance(msg, HumanMessage) else "ai"

    with st.chat_message(role):
        st.markdown(msg.content)
