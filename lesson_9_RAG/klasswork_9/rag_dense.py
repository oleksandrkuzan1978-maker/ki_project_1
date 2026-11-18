import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3

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


def mean_pooling(model_output, attention_mask):
    """Усредняем токены, исключая padding."""
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


def retrieve(query, tokenizer, model):
    passages = [item["text"] for item in KNOWLEDGE_BASE]
    doc_vectors = encode(passages, tokenizer, model, "passage: ")
    query_vector = encode([query], tokenizer, model, "query: ")

    scores = (query_vector @ doc_vectors.T).squeeze(0)
    best_indices = torch.topk(scores, k=min(TOP_K, len(KNOWLEDGE_BASE))).indices.tolist()

    return [
        (KNOWLEDGE_BASE[i], float(scores[i]))
        for i in best_indices
    ]


def main():
    query = 'код протокола EXT-GW-91B7'
    tokenizer, model = load_model()
    results = retrieve(query, tokenizer, model)

    print("\n=== DENSE RAG: только векторный поиск ===")
    print(f"Запрос: {query}\n")
    print("Найденные фрагменты:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n{rank}. {doc['id']} | cosine similarity = {score:.4f}")
        print(f"   {doc['text']}")

    context = "\n".join(doc["text"] for doc, _ in results)
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
