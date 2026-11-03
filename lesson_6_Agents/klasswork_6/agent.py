
# Подключаем модель, инструменты, сообщения, память агента и загрузку настроек.
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_community.tools.tavily_search import TavilySearchResults  
from langchain_core.messages import HumanMessage  
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.prebuilt import create_react_agent  
from dotenv import load_dotenv  
import os  


# Загружаем настройки и API-ключи из файла .env в переменные окружения.
load_dotenv()


# Передаём ключ Gemini под именем GOOGLE_API_KEY, которое ожидает интеграция.
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# Передаём ключ Tavily для доступа к интернет-поиску.
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# Создаём подключение к модели Gemini; запросы отправятся при запуске агента.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Храним состояние диалога в оперативной памяти, пока работает программа.
memory = MemorySaver()


# Создаём инструмент интернет-поиска: до двух результатов на один вызов.
search = TavilySearchResults(max_results=2)


# Передаём агенту доступные инструменты: поиск.
tools = [search]



# Собираем ReAct-агента: модель выбирает инструменты и использует их ответы.
# Checkpointer сохраняет состояние диалога между обращениями.
agent_executor = create_react_agent(llm, tools, checkpointer=memory)


# Идентификатор диалога: одинаковый thread_id позволяет продолжить его историю.
config = {"configurable": {"thread_id": "abc123"}}


# Первое обращение: сообщаем имя и город пользователя.
# stream_mode="values" выдаёт полное состояние после каждого шага агента.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in berlin")]},
    config,
    stream_mode="values",
):
    
    # Выводим последнее сообщение текущего состояния: пользователя, модели или инструмента.
    step["messages"][-1].pretty_print()



# Продолжаем тот же диалог: агенту доступен ранее названный город — Берлин.
# Для ответа о погоде модель может вызвать инструмент поиска.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    
    # Выводим последнее сообщение текущего состояния: пользователя, модели или инструмента.
    step["messages"][-1].pretty_print()