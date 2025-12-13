import os
import dotenv
from langchain_google_genai import GoogleGenerativeAI


dotenv.load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

llm = GoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    top_k=3,
    top_p=0.9,
    temperature=0
)

with open('return_policy.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

print(source_data)

while True:
    user_input = input('Ваше питання щодо умов повернення товару: ').strip()

    if not user_input:
        break

    instruction = f'відповідай лише на питання щодо умов повернення товару, якщо користувач запитує не по темі, сухо відповідай, що питання не стосується умов повернення товару". '\
                  f'Відповідай мовою, якою написане це речення:"{user_input}".'

    response = llm.invoke(f'{source_data}\n{instruction}\n{user_input}')

    print(response)