# Завдання 1
# Напишіть модель для рекомендації книг з двох ланцюгів
#  Перший ланцюг отримує назву книги та визначає її жанр
#  Другий отримує назву книги, жанр та повертає список
# схожих книг(того ж самого жанру та іншого)

import os
import dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


dotenv.load_dotenv()
API_KEY = os.getenv(GEMINI_API_KEY)

llm = GoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=API_KEY,
    temperature=0
)

class BookInfo(BaseModel)
    genre str = Field(description='жанр книги')

parser = PydanticOutputParser(pydantic_object=BookInfo)

instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""Ти - бібліотекар. Твоє завдання полягає в тому, щоб визначати жанри книг.
### ІСТРУКЦІЇ
{instructions}

### НАЗВА КНИГИ
{book_title}"""
, partial_variables={'instructions': instructions})

chain1 = prompt | llm | parser

user_book = input('Введіть назву книги ')

response = chain1.invoke({
    'book_title': user_book
})

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