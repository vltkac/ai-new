import os
import dotenv
from typing import List
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
)

messages: List[BaseMessage] = [
    SystemMessage(
        """
Ти - всезнаючий чат, який може цікаво розповідати про все, що існує у світі.
"""
    )
]

summary_prompt = PromptTemplate.from_template(
    """
Ти - помічник для підсумовування інформації.
Тобі нададуть історію чату.
Потрібно зробити коротку сводку та зберегти якомога більше важливих деталей.
Роби лише підсумок тексту без зазначення, хто саме що писав.

# ІСТОРІЯ ЧАТУ
{history}
"""
)

summary_chain = summary_prompt | llm

chat_history_text = ""

while True:
    user_query = input("Ваше повідомлення: ")
    if user_query == "":
        break

    messages.append(HumanMessage(user_query))
    chat_history_text += f"{user_query}\n"

    response = llm.invoke(messages)
    messages.append(response)
    chat_history_text += f"{response.content}\n"

    print(f"AI: {response.content}")

    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) > 4:
        summary = summary_chain.invoke({"history": chat_history_text})
        messages = [messages[0], AIMessage(summary.content)]
        chat_history_text = summary.content + "\n"