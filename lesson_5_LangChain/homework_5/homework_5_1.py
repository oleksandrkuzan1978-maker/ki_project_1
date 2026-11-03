from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os
import textwrap

user_url = 'https://izum.ua/ru/hto-pereklav-bibliyu-na-ukrayinsku-movu-zahoplyva-istoriya-perekladiv/'

# Загружаем ключ API из переменных окружения.
load_dotenv()
api_key =os.getenv('GEMINI_API_KEY')

# Подключаем модель Gemini.
llm = ChatGoogleGenerativeAI(
                            model='gemini-3.5-flash-lite'
                             , google_api_key=api_key
                             )

# Загружаем текст веб-страницы.
def get_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

# Создаём инструкцию для суммаризации.
prompt = ChatPromptTemplate.from_template("Напишите краткое изложение следующего текста: {context}")

# Собираем цепочку из модели и шаблона запроса.
chain = create_stuff_documents_chain(llm, prompt)

# Запускаем обработку.

result = chain.invoke({"context": get_docs(user_url)})

for paragraph in result.splitlines():
    print(textwrap.fill(paragraph, width=120))