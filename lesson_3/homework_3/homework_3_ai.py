import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.5-flash-lite")

response = model.generate_content("Explain what Python is in two sentences.")

res = response.text
print(type(res))
res = res.split(".", 1)
res[0] = res[0]+"."
for sentence in res:
    print(sentence)

