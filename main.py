# Напишіть модель для генерації резюме:
#  Перший ланцюг отримує опис вакансії та повертає
# основні навички, які необхідні
#  Другий ланцюг отримує основні навички та опис
# кандидата і генерує резюме

import os
import dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()
API_KEY = os.getenv(GEMINI_API_KEY)

llm = GoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=API_KEY,
    temperature=0
)

class Quality(BaseModel):
    quals: List[str] = Field(description='свойства, которые нужны для вакансии')

parser_1 = PydanticOutputParser(pydantic_object=Quality)
guide_1 = parser_1.get_format_instructions()

prompt = PromptTemplate.from_template("""
Ты - HR-специалист. Твоя задача - передавать список основных качеств человека, которые необходимы для вакансии.

### ИНСТРУКЦИИ
{guide}

### ВАКАНСИЯ
{position}
""", partial_variables={'guide': guide_1})

chain_1 = prompt | llm | parser_1

response_1 = chain_1.invoke({
    'guide': "Программист"
})

<<<<<<< HEAD
class UpgradedBookInfo(BaseModel):
    books: List[str] = Field(description='список схожих книг')

parser = PydanticOutputParser(pydantic_object=UpgradedBookInfo)

instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""Ти - бібліотекар. Твоє завдання полягає в тому, щоб радити подібні книги на основі назви книги та жанру.
### ІСТРУКЦІЇ
{instructions}

### НАЗВА КНИГИ
{book_title}

### ЖАНР
{book_genre}""", partial_variables={'instructions': instructions})

chain1 = prompt | llm | parser

full_response = chain1.invoke({
    'book_title': user_book,
    'book_genre': response.genre
})

for book in full_response.books:
    print(book)
=======
print(response_1.quals)
>>>>>>> aa7f7a2 (.)
