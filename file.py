import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(api_key[:5])

genai.configure(api_key=api_key)

#model = genai.GenerativeModel("gemini-2.5-flash")
model = genai.GenerativeModel("gemini-3.8-flash")

#response = model.generate_content("почему при одинаковой архитектуре llm и одинаковом промпте всегда генерируются разные токены и разный ответ?")
response = model.generate_content("какой сегодня день недели, месяц, год?")
print(response.text)


# prompt = 'what is the weather in Berlin? {}'


# add_info = 'windy, sunny.'

# final_prompt = prompt.format(add_info)

# print(prompt)
# print(final_prompt)


