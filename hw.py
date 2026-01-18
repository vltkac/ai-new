import os
import dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("soup")

store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)


def search_terms(query: str):
    return store.similarity_search(query, k=5)


agent = create_react_agent(
    model=llm,
    tools=[search_terms],
)

dialog = [
    SystemMessage(
        """
        Ты выступаешь как справочный ассистент по правилам и условиям сервисов Google.
        Отвечай только на основе найденных фрагментов.
        Всегда используй инструмент поиска перед формированием ответа.
        Запрос пользователя сначала приводи к украинскому языку для поиска,
        но финальный ответ давай на языке запроса.
        Если информации недостаточно, прямо сообщи об этом.

        Доступный инструмент:
        - search_terms
        """
    )
]

while True:
    text = input("Ви: ")
    if not text:
        break

    dialog.append(HumanMessage(text))

    result = agent.invoke({"messages": dialog})
    dialog = result["messages"]

    reply = dialog[-1]
    print(reply.content)
    print("\nІсторія")

    for msg in dialog:
        print(repr(msg))