# PyTorch выполняет вычисления нейросети и работает с тензорами — многомерными массивами.
import torch
# Классы для загрузки токенизатора и готовой языковой модели.
from transformers import AutoTokenizer, AutoModel
# NumPy нужен для вычислений с векторами.
import numpy as np


# Выбираем многоязычную модель DistilBERT, учитывающую регистр букв.
model_name = "distilbert-base-multilingual-cased"


# Загружаем токенизатор: он разбивает текст на токены и присваивает им номера.
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Загружаем модель; при отсутствии в локальном кеше файлы скачиваются.
model = AutoModel.from_pretrained(model_name)


# Включаем режим обработки данных: отключаем случайный dropout, используемый при обучении.
model.eval()


# Три английских слова для сравнения их векторных представлений.
text1 = 'car'
text2 = 'building'
text3 = 'nature'

# Возвращаем читаемые токены и набор тензоров для подачи в модель.
def get_tokens(string):
    # Разбиваем строку на токены, включая специальные служебные токены.
    tokens = tokenizer(
        string,
        # Возвращаем данные в виде тензоров PyTorch.
        return_tensors="pt"
    )
    # Преобразуем номера первой строки в токены; также возвращаем все входные данные.
    return tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]), tokens

# output, tokens = get_tokens(text1)
# print(output)
# print(tokens)
# print("Tokens:")
# print(get_tokens(text1))

# Пример фразы, на которой покажем результат токенизации.
input_string = 'i love programming'
result = get_tokens(input_string)
# Печатаем токены и тензоры, полученные для примера.
print(result)

# Берём второй элемент результата — входные тензоры; повторяем для остальных слов.
tokens1 = get_tokens(text1)[1]
tokens2 = get_tokens(text2)[1]
tokens3 = get_tokens(text3)[1]

# Получаем отдельный вектор для каждого токена входного текста.
def get_embedding(tokens):
    # Не сохраняем данные для вычисления градиентов: обучение здесь не выполняется.
    with torch.no_grad():
        # Передаём элементы словаря как именованные аргументы модели.
        outputs = model(**tokens)

    # Векторы последнего слоя: размерность [число текстов, число токенов, длина вектора].
    embeddings = outputs.last_hidden_state
    # Возвращаем тензор с векторами всех токенов.
    return embeddings


# emb = get_embedding(tokens)[0]
# print('<----->')
# print(emb)
# print(emb.shape)

# Закомментированный пример: усредняет векторы токенов в один вектор текста.
# def get_sentence_embedding(tokens):
# Маска отмечает реальные токены единицей, а заполнители для выравнивания длины — нулём.
#     attention_mask = tokens["attention_mask"]

#     with torch.no_grad():
#         outputs = model(**tokens)

#     token_embeddings = outputs.last_hidden_state

# Добавляем измерение, чтобы маску можно было умножить на векторы токенов.
#     mask = attention_mask.unsqueeze(-1)

# Суммируем векторы по маске и делим на число учтённых токенов; служебные токены тоже учитываются.
#     sentence_embedding = (
#         token_embeddings * mask
#     ).sum(dim=1) / mask.sum(dim=1)
#     return sentence_embedding

# print("\nEmbeddings shape:")
# Берём первый текст и токен с индексом 1 (после начального служебного), затем переводим в NumPy.
emb1 = get_embedding(tokens1)[0, 1].numpy()
# Аналогично получаем вектор первого содержательного токена второго слова.
emb2 = get_embedding(tokens2)[0, 1].numpy()
# Вектор первого содержательного токена третьего слова; это не усреднение всей фразы.
emb3 = get_embedding(tokens3)[0, 1].numpy()

# print(tokens1)
# print(emb1)

# Косинусное сходство сравнивает направления ненулевых векторов: от -1 до 1.
def get_similarity(vector1, vector2):
    # Делим скалярное произведение на произведение длин: ближе к 1 — более схожее направление.
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

# Сравниваем car с building.
cos12 = get_similarity(emb1, emb2)
# Сравниваем car с nature.
cos13 = get_similarity(emb1, emb3)

# Простой числовой пример: у векторов [2, 2] и [5, 5] одинаковое направление.
v1 = np.array([2, 2])
v2 = np.array([5, 5])
# Их косинусное сходство равно 1, хотя длины различаются.
cosv1v2 = get_similarity(v1, v2)
# Эти строки вывода отключены; уберите начальный #, чтобы увидеть оценки.
# print(cos12)
# print(cos13)
# print(cosv1v2)


# sentence_emb1 = get_sentence_embedding(tokens1)
# print(sentence_emb1)