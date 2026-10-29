import os
import numpy as np
import faiss

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.retrievers.kendra import combined_text

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Не найден GEMINI_API_KEY в файле .env")

genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 768


texts_to_index = [
    "Кошка сидит на окне",
    "Собака играет в парке",
    "Ананас растет в тропиках",
    "Кот спит на диване",
    "Пес лает на почтальона",
    "Фрукт ананас очень вкусный",
    "Домашняя кошка любит ласку",
    "Верный пес охраняет дом",
    "Спелый ананас полон витаминов",
    "Кошки любят рыбу и мясо",
    "Основной рацион кошек - это белок",
    "Чем кормить котенка?",
    "Лучший корм для кошек - сбалансированный",
    "Коты едят сухой и влажный корм",
    "Нельзя кормить кошку шоколадом",
    "Молоко не всегда полезно для кошек",
    "Я обожаю программировать.",
    "Программирование – это то, что мне очень нравится.",
    "Меня увлекает разработка программного обеспечения.",
    "Я испытываю страсть к написанию кода.",
    "Программирование приносит мне огромное удовольствие.",
    "Мне интересно заниматься программированием.",
    "Моё хобби – это кодинг.",
    "В свободное время я занимаюсь программированием.",
    "Кодинг – это моё любимое увлечение.",
    "Я увлекаюсь кодингом на досуге.",
    "Программирование – это моё хобби и страсть.",
    "Когда есть свободное время, я кодирую.",
    "Кодинг - это моё хобби, которым я наслаждаюсь.",
    "Маленький котенок играет с клубком ниток.",
    "Домашние питомцы нуждаются в заботе и внимании.",
    "Собака радостно бегает за мячом.",
    "Ананас обладает сладким вкусом и приятным ароматом.",
    "Тропические фрукты богаты полезными веществами.",
    "Правильное питание помогает кошкам оставаться здоровыми.",
    "Я люблю создавать приложения на Python.",
    "Разработка программ стала моим любимым занятием.",
    "Вечерами я пишу код для собственных проектов.",
    "Изучение новых языков программирования доставляет мне радость.",
    "Создание программного обеспечения требует логического мышления."
]


def get_embedding(text, task_type):
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIMENSION
    )

    return np.asarray(
        response["embedding"],
        dtype=np.float32
    )

def build_index(texts):
    embeddings = []

    for i, text in enumerate(texts, start=1):
        print(f"Индексация {i}/{len(texts)}: {text}")

        vector = get_embedding(
            text,
            task_type="RETRIEVAL_DOCUMENT"
        )

        embeddings.append(vector)

    embeddings_array = np.ascontiguousarray(
        np.stack(embeddings),
        dtype=np.float32
    )
    faiss.normalize_L2(embeddings_array)

    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

    return index


def semantic_search(query, index, texts, k=4):
    query_embedding = get_embedding(
        query,
        task_type="RETRIEVAL_QUERY"
    )

    query_array = np.ascontiguousarray(
        query_embedding.reshape(1, -1),
        dtype=np.float32
    )

    faiss.normalize_L2(query_array)

    k = min(k, index.ntotal)

    scores, indices = index.search(query_array, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "text": texts[idx],
            "similarity": float(score)
        })

    return results


def get_search_query():
    return input("Введите поисковый запрос: ")


if __name__ == "__main__":
    index = build_index(texts_to_index)

    print("\nРазмерность индекса:", index.d)
    print("Количество векторов:", index.ntotal)

    search_query = get_search_query() # "Люблю писать код"

    results = semantic_search(
        search_query,
        index,
        texts_to_index,
        k=4
    )

    print("\n--- Результаты семантического поиска ---")
    print(f"Запрос: {search_query}")

    combined_text = " ".join(result["text"] for result in results)
    print("\nОбъединенный текст:")
    print(combined_text)




    #
    # for result in results:
    #     print(
    #         f"- {result['text']} "
    #         f"(similarity: {result['similarity']:.4f})"
    #     )