# LLM
# Large language model

# загрузка API ключа с ключа .env как переменную среды

# import os
# import dotenv
#
#
# dotenv.load_dotenv()
# api_key = os.getenv('GEMINI_API_KEY')
#
# import langchain
# from langchain_google_genai import GoogleGenerativeAI
#
#
# llm = GoogleGenerativeAI(
#     model='gemini-2.5-flash-lite',
#     api_key=api_key
# )
#
# response = llm.invoke('Hello, what are your salary expectations?')
# print(response)


# Завдання 1
# Підключіть модель LLM за допомогою свого API key.
# Попросіть модель згенерувати:
# ● відповідь на питання у вигляді одного
# слова(наприклад яка столиця Франції?)
# ● код python
# ● коротку історію
# Підберіть параметри креативності та довжини

import os
import dotenv
from langchain_google_genai import GoogleGenerativeAI


dotenv.load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

llm = GoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    temperature=0
)

user_input = input('Your question: ')
# command1 = 'дай ответ одним словом. если ответ два и больше слова, то давай полный ответ. '
# command_py = 'write response of the python code (only): '
command_story = 'write story within 4 sentences. be creative and fun. '

response = llm.invoke(command_story + user_input)
print(response)
















