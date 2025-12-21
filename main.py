import os
import dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

dotenv.load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0
)

# ---------- ПЕРШИЙ ЛАНЦЮГ ----------

class ExerciseList(BaseModel):
    exercises: List[str] = Field(description="список вправ відповідно до мети тренувань")

parser_1 = PydanticOutputParser(pydantic_object=ExerciseList)
instructions_1 = parser_1.get_format_instructions()

prompt_1 = PromptTemplate.from_template(
    """Ти — професійний фітнес-тренер.
На основі мети тренувань підбери відповідний список вправ.

### ІНСТРУКЦІЇ
{instructions}

### МЕТА ТРЕНУВАНЬ
{target}
""",
    partial_variables={"instructions": instructions_1}
)

chain_1 = prompt_1 | llm | parser_1

user_target = input("Введіть мету тренувань: ")

exercise_response = chain_1.invoke({
    "target": user_target
})

# ---------- ДРУГИЙ ЛАНЦЮГ ----------

class TrainingPlan(BaseModel):
    plan: str = Field(description="детальний план тренувань на тиждень")

parser_2 = PydanticOutputParser(pydantic_object=TrainingPlan)
instructions_2 = parser_2.get_format_instructions()

prompt_2 = PromptTemplate.from_template(
    """Ти — персональний фітнес-інструктор.
Склади тижневий план тренувань на основі вправ,
рівня підготовки та доступного часу.

### ІНСТРУКЦІЇ
{instructions}

### СПИСОК ВПРАВ
{exercises}

### РІВЕНЬ ПІДГОТОВКИ
{level}

### ЧАС НА ТИЖДЕНЬ (в годинах)
{hours}
""",
    partial_variables={"instructions": instructions_2}
)

chain_2 = prompt_2 | llm | parser_2

user_level = input("Введіть рівень підготовки (низький / середній / професіонал): ")
user_hours = input("Введіть кількість годин на тиждень: ")

training_plan_response = chain_2.invoke({
    "exercises": ", ".join(exercise_response.exercises),
    "level": user_level,
    "hours": user_hours
})

print("\nПлан тренувань:\n")
print(training_plan_response.plan)