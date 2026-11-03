# Подключаем инструменты LangChain: сборку цепочки из документов,
# шаблон диалога и обёртку для использования обычной Python-функции в цепочке.
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Загрузчик извлекает текст веб-страницы и возвращает документы LangChain.
from langchain_community.document_loaders import WebBaseLoader

# Transformers загружает токенизатор и языковую модель, а PyTorch выполняет вычисления.
# Токенизатор переводит текст в числовые коды (токены) и обратно.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Выбираем готовую модель Qwen, обученную выполнять инструкции.
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Загружаем токенизатор выбранной модели. При первом запуске файлы могут
# скачиваться из Hugging Face, затем используются сохранённые локальные копии.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Загружаем веса модели. float32 задаёт формат чисел для её вычислений.
# В этом коде модель явно не переносится на видеокарту.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

# Переключаем модель в режим получения ответов: отключаем поведение,
# которое нужно только при обучении, например случайное отключение нейронов.
model.eval()


# 2. Описываем, как передать запрос LangChain локальной модели и получить ответ.
def generate_text(prompt_value):
    # Превращаем подготовленный запрос LangChain в список сообщений диалога.
    messages = prompt_value.to_messages()

    # Здесь соберём сообщения в формате, который ожидает токенизатор модели.
    hf_messages = []

    for message in messages:
        # Сопоставляем названия ролей двух библиотек:
        # system — инструкции, user — запрос пользователя, assistant — ответ модели.
        # Все остальные типы сообщений этот код также считает ответами assistant.
        if message.type == "system":
            role = "system"
        elif message.type == "human":
            role = "user"
        else:
            role = "assistant"

        # Для каждого сообщения сохраняем роль и сам текст.
        hf_messages.append({
            "role": role,
            "content": message.content
        })

    # Оформляем диалог по правилам Qwen и сразу переводим его в токены.
    # add_generation_prompt добавляет обозначение начала ответа модели.
    # return_tensors="pt" возвращает данные в формате PyTorch.
    # truncation ограничивает вход 6000 токенами: лишняя часть будет отброшена,
    # поэтому слишком длинная страница попадёт в запрос не целиком.
    inputs = tokenizer.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=6000
    )

    # Для получения ответа не нужны градиенты (данные для обучения).
    # Их отключение сокращает расход памяти при вычислениях.
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=250,  # Генерируем не более 250 новых токенов.
            do_sample=False,  # Выбираем продолжение без случайного сэмплирования.
            repetition_penalty=1.15,  # Снижаем склонность повторять уже выданные токены.
            no_repeat_ngram_size=4,  # Запрещаем повтор одинаковых цепочек из 4 токенов.
            pad_token_id=tokenizer.eos_token_id  # Используем код конца текста для заполнения.
        )

    # Результат содержит исходный запрос и продолжение. Берём первый ответ
    # и отрезаем входные токены, чтобы оставить только сгенерированный текст.
    generated_tokens = outputs[0][inputs.shape[-1]:]

    # Превращаем токены ответа в строку, убирая служебные обозначения модели.
    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


# Оборачиваем функцию в объект LangChain, который можно включить в цепочку.
llm = RunnableLambda(generate_text)


# 3. Описываем загрузку текста по адресу страницы.
def get_docs(url):
    # Создаём загрузчик для указанного URL.
    loader = WebBaseLoader(url)
    # Получаем список документов; текст каждого хранится в поле page_content.
    return loader.load()


# 4. Готовим шаблон запроса. Системное сообщение задаёт правила ответа,
# а в пользовательское сообщение вместо {context} будет подставлен текст документов.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты помощник, специализирующийся на суммаризации текстов. "
        "Отвечай исключительно на русском языке. "
        "Не рассказывай о себе. Не используй шаблонные вступления. "
        "Составь краткое и информативное изложение текста."
    ),
    (
        "human",
        "Сделай краткое изложение следующего текста:\n\n{context}"
    )
])


# 5. Собираем цепочку: она объединит тексты документов в один контекст,
# вставит его в шаблон и вызовет нашу функцию генерации через llm.
chain = create_stuff_documents_chain(llm, prompt)


# 6. Получаем адрес страницы от пользователя и загружаем её текст.
user_url = input("Введите адрес веб-страницы: ")

docs = get_docs(user_url)

# Показываем объём загруженных данных. Размер здесь измеряется в символах,
# а не в токенах; это объём до ограничения входа внутри generate_text.
print("Количество документов:", len(docs))
print("Размер текста:", sum(len(doc.page_content) for doc in docs))

# 7. Запускаем цепочку. Ключ context передаёт список документов,
# из которых цепочка подготовит текст для краткого изложения.
result = chain.invoke({
    "context": docs
})

# 8. Выводим полученное краткое изложение в консоль.
print("\nРЕЗУЛЬТАТ СУММАРИЗАЦИИ:\n")
print(result)
