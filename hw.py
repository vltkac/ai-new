import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_api_key,
)

searcher = GoogleSerperAPIWrapper(
    serper_api_key=serper_api_key,
    type="places",
)

def get_places_info(query: str) -> list:
    search_info = searcher.results(query)
    places = search_info.get("places", [])
    result = []

    for place in places:
        info = {
            "title": place.get("title"),
            "website": place.get("website"),
            "rating": place.get("rating"),
        }
        result.append(info)

    return result

agent = create_react_agent(
    model=llm,
    tools=[get_places_info],
)

messages = [
    SystemMessage(
        """
Ти - консультант з рекомендацій ресторанів та закладів харчування.
Користувачі будуть запитувати про ресторани в різних містах та країнах.
Твоя задача - рекомендувати ресторани, використовуючи інструмент пошуку місць.
Завжди отримуй дані через інструмент.
Якщо частини інформації не вистачає - логічно доповни відповідь.
Описуй ресторани привабливо та зрозуміло.
Перед використанням інструменту перекладай запит користувача англійською.
Якщо запит нечіткий - уточни деталі.
"""
    )
]

while True:
    user_query = input("Ви: ")
    if user_query == "":
        break

    messages.append(HumanMessage(user_query))

    response = agent.invoke({"messages": messages})
    messages = response["messages"]

    print(messages[-1].content)