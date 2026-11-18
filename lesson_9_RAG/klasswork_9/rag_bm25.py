import math
import re
import sys
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3
RRF_K = 60

KNOWLEDGE_BASE = [
    {
        "id": "doc_01",
        "text": "В компании действует политика удалённой работы: сотрудники могут работать из дома до трёх дней в неделю. Для этого необходимо согласовать график с непосредственным руководителем."
    },
    {
        "id": "doc_02",
        "text": "Компенсация за использование личного оборудования предоставляется сотрудникам в размере 1500 рублей в месяц. Для получения выплаты нужно подать заявку в HR-портал до пятого числа."
    },
    {
        "id": "doc_03",
        "text": "Стандартный срок хранения резервных копий рабочих баз данных составляет 30 календарных дней. Критичные системы дополнительно копируются ежедневно."
    },
    {
        "id": "doc_04",
        "text": "Отпуск необходимо запрашивать через корпоративный портал не позднее чем за 14 календарных дней до его начала. Заявка считается одобренной только после подтверждения руководителем."
    },
    {
        "id": "doc_05",
        "text": "Для доступа к VPN требуется многофакторная аутентификация. Одноразовый код приходит в приложение-аутентификатор, а не по электронной почте."
    },
    {
        "id": "doc_06",
        "text": "Если ноутбук сотрудника сломался, нужно создать обращение в службу технической поддержки. При критической неисправности следует позвонить на внутренний номер 404."
    },
    {
        "id": "doc_07",
        "text": "Бюджет на обучение составляет до 40 000 рублей на сотрудника в год. Курс должен быть связан с текущей ролью или индивидуальным планом развития."
    },
    {
        "id": "doc_08",
        "text": "Пароли корпоративных учётных записей должны содержать не менее 14 символов. Запрещено повторно использовать последние пять паролей."
    },
    {
        "id": "doc_09",
        "text": "Переговорные комнаты бронируются в календаре компании. Бронирование автоматически отменяется, если встреча не началась в течение 15 минут."
    },
    {
        "id": "doc_10",
        "text": "Расходы на деловые поездки возмещаются после загрузки чеков и отчёта о поездке в систему Expense. Обычно обработка занимает до десяти рабочих дней."
    },
    {
        "id": "doc_11",
        "text": "При возникновении ошибки ERR_AUTH_4037 необходимо проверить срок действия токена авторизации. Если токен истёк, следует выполнить повторную аутентификацию."
    },
    {
        "id": "doc_12",
        "text": "Модуль сетевого адаптера с артикулом NIC-X710-DA2 поддерживает два порта 10GbE и используется в серверах серии Enterprise R740."
    },
    {
        "id": "doc_13",
        "text": "Начиная с версии libcrypto 3.2.7, исправлена уязвимость обработки сертификатов с некорректным расширением Subject Alternative Name."
    },
    {
        "id": "doc_14",
        "text": "Инцидент INC-2025-08421 был вызван переполнением очереди RabbitMQ. Для устранения проблемы необходимо увеличить параметр consumer_prefetch_count до 50."
    },
    {
        "id": "doc_15",
        "text": "Для устранения таймаутов PostgreSQL необходимо установить параметр statement_timeout=45000 в конфигурации сервиса аналитики."
    },
        {
        "id": "doc_16",
        "text": "Для восстановления доступа к системе необходимо применить аварийный ключ восстановления KX-9941-ZQ. Ключ хранится в защищённом хранилище инфраструктурной команды."
    },
    {
        "id": "doc_17",
        "text": "В конфигурации сервиса платежей установлен параметр PAYMENT_RETRY_JITTER=0.037. Изменение данного значения допускается только после согласования с командой разработки."
    },
    {
        "id": "doc_18",
        "text": "Сервер с инвентарным номером SRV-88291-A располагается в стойке R14, дата-центре DC-7. На сервере размещена база данных резервного копирования."
    },
    {
        "id": "doc_19",
        "text": "Для подключения к тестовому API необходимо использовать заголовок X-ORBIT-TRACE-KEY со значением ORB-71F9-C2. Без данного заголовка сервер возвращает HTTP 401."
    },
    {
        "id": "doc_20",
        "text": "В системе мониторинга зарегистрирован алерт ALR-993821. Причиной срабатывания стало превышение параметра max_pending_jobs выше значения 187."
    },
    {
        "id": "doc_21",
        "text": "Микросервис обработки документов использует очередь с идентификатором Q-DOC-771-X. Для проверки состояния очереди необходимо обратиться к административному интерфейсу RabbitMQ."
    },
    {
        "id": "doc_22",
        "text": "Для интеграции с внешним шлюзом используется код протокола EXT-GW-91B7. Он определяет формат передачи бинарных сообщений и правила подтверждения доставки."
    },
    {
        "id": "doc_23",
        "text": "В production-конфигурации Kubernetes для namespace analytics-prod-73 установлено значение memory_limit=768Mi. Увеличение лимита требует согласования с DevOps."
    },
    {
        "id": "doc_24",
        "text": "Ошибка драйвера устройства зарегистрирована под кодом DEV-FAULT-8C291. Для устранения неисправности необходимо обновить firmware до версии 6.18.42."
    },
    {
        "id": "doc_25",
        "text": "Внутренний регламент обработки транзакций имеет идентификатор REG-FIN-2026-0049. Документ определяет порядок повторной обработки транзакций со статусом PENDING_REVIEW."
    }
]


def tokenize(text):
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def encode(texts, tokenizer, model, prefix):
    inputs = tokenizer(
        [prefix + text for text in texts],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def bm25_scores(query, documents, k1=1.5, b=0.75):
    tokenized_docs = [tokenize(doc) for doc in documents]
    query_terms = tokenize(query)
    doc_count = len(tokenized_docs)
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lengths) / max(doc_count, 1)

    document_frequency = Counter()
    term_frequencies = []
    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequency.update(set(tokens))

    scores = [0.0] * doc_count
    for term in query_terms:
        df = document_frequency.get(term, 0)
        if df == 0:
            continue

        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

        for i, frequencies in enumerate(term_frequencies):
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            denominator = tf + k1 * (
                1 - b + b * doc_lengths[i] / max(avgdl, 1e-9)
            )
            scores[i] += idf * (tf * (k1 + 1)) / denominator

    return scores


def dense_scores(query, tokenizer, model):
    passages = [item["text"] for item in KNOWLEDGE_BASE]
    doc_vectors = encode(passages, tokenizer, model, "passage: ")
    query_vector = encode([query], tokenizer, model, "query: ")
    return (query_vector @ doc_vectors.T).squeeze(0).tolist()


def rank_indices(scores):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def retrieve(query, tokenizer, model):
    documents = [item["text"] for item in KNOWLEDGE_BASE]

    dense = dense_scores(query, tokenizer, model)
    bm25 = bm25_scores(query, documents)

    dense_rank = rank_indices(dense)
    bm25_rank = rank_indices(bm25)

    fused = [0.0] * len(KNOWLEDGE_BASE)
    for ranking in (dense_rank, bm25_rank):
        for rank, doc_index in enumerate(ranking, start=1):
            fused[doc_index] += 1.0 / (RRF_K + rank)

    final_rank = rank_indices(fused)[:TOP_K]
    return [
        (KNOWLEDGE_BASE[i], dense[i], bm25[i], fused[i])
        for i in final_rank
    ], dense_rank, bm25_rank


def main():
    query = 'код протокола EXT-GW-91B7'
    tokenizer, model = load_model()
    results, dense_rank, bm25_rank = retrieve(query, tokenizer, model)

    print("\n=== HYBRID RAG: dense + BM25 + RRF ===")
    print(f"Запрос: {query}\n")
    print("Результаты итогового объединённого ранжирования:")
    for rank, (doc, dense, bm25, fused) in enumerate(results, start=1):
        print(
            f"\n{rank}. {doc['id']} | dense={dense:.4f} "
            f"| BM25={bm25:.4f} | RRF={fused:.6f}"
        )
        print(f"   {doc['text']}")

    print("\nПозиции документов в отдельных списках (сверху — лучше):")
    print("Dense:", " > ".join(KNOWLEDGE_BASE[i]["id"] for i in dense_rank[:5]))
    print("BM25: ", " > ".join(KNOWLEDGE_BASE[i]["id"] for i in bm25_rank[:5]))

    context = "\n".join(doc["text"] for doc, _, _, _ in results)
    prompt = (
        "Ответь на вопрос, используя только контекст. Если ответа в контексте нет, "
        "скажи об этом.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}"
    )
    print("\n=== Контекст, передаваемый генеративной модели ===")
    print(prompt)


if __name__ == "__main__":
    main()
