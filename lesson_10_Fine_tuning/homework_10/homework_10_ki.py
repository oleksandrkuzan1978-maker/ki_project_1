"""Дообучение DistilBERT на синтетических обращениях в интернет-магазин."""
import argparse
import unicodedata

import torch
from transformers import AutoModel, AutoTokenizer


# Метка -> обучающие тексты и отдельные тестовые формулировки.
DATA = {
    "доставка": (
        [
            "Когда привезут заказ?", "Где сейчас моя посылка?",
            "Курьер сегодня не приехал.", "Измените адрес доставки.",
            "Можно перенести доставку на завтра?", "Не вижу номера отслеживания.",
            "Заказ ещё в пути?", "В какой пункт выдачи придёт посылка?",
            "Отправили ли вы мою покупку?", "Перенесите визит курьера на вечер.",
            "Сколько ждать отправления?", "Посылка застряла на сортировке.",
            "  Когда   привезут заказ?  ", "",
        ],
        [
            "В какой день ожидать курьера?",
            "По трек-номеру давно нет движения.",
            "Хочу забрать заказ в другом постамате.",
            "Моя коробка до сих пор не пришла.",
        ],
    ),
    "оплата": (
        [
            "Не проходит оплата картой.", "Деньги списали два раза.",
            "Где скачать чек об оплате?", "Можно заплатить при получении?",
            "Банк отклонил платёж.", "На странице оплаты возникла ошибка.",
            "Принимаете ли вы банковский перевод?", "Как сменить способ оплаты?",
            "Нет подтверждения платежа.", "Могу оплатить заказ частями?",
            "С меня списали лишнюю сумму.", "Не получается ввести данные карты.",
            "не проходит оплата картой.",
        ],
        [
            "За покупку взяли деньги дважды.",
            "При расчёте сайт выдаёт ошибку.",
            "Где найти квитанцию за покупку?",
            "Доступна ли рассрочка?",
        ],
    ),
    "возврат товара": (
        [
            "Как оформить возврат товара?", "Размер не подошёл, хочу отправить обратно.",
            "Куда отнести вещь для возврата?", "Прислали брак, хочу вернуть покупку.",
            "Какой срок возврата обуви?", "Нужна наклейка для обратной отправки.",
            "Можно ли вернуть нераспакованную вещь?", "Отмените заявку на возврат.",
            "Как вернуть только один товар из заказа?", "Примете ли товар без упаковки?",
            "Получили ли вы отправленный обратно товар?", "Хочу обменять платье на другой размер.",
            "Как   оформить возврат товара?",
        ],
        [
            "Кофта оказалась мала. Могу её сдать?",
            "Товар с дефектом, хочу отдать его обратно.",
            "Сколько дней у меня есть на обратную отправку?",
            "Подскажите, где получить этикетку для возврата.",
        ],
    ),
}


def prepare_data():
    train, test = {}, {}
    raw_count = 0
    for label, (train_texts, test_texts) in enumerate(DATA.values()):
        for texts, split in ((train_texts, train), (test_texts, test)):
            for text in texts:
                raw_count += 1
                cleaned = " ".join(unicodedata.normalize("NFKC", text).split())
                if cleaned:
                    key = cleaned.casefold()
                    previous = split.setdefault(key, (cleaned, label))
                    if previous[1] != label:
                        raise ValueError(f"Противоречивые метки: {cleaned}")
    if train.keys() & test.keys():
        raise ValueError("Дубликаты между обучением и тестом")
    return list(train.values()), list(test.values()), raw_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilbert-base-multilingual-cased")
    parser.add_argument("--save-dir", help="Куда сохранить обученную модель")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)

    train, test, raw_count = prepare_data()
    labels = list(DATA)
    print(f"Исходных строк: {raw_count}. Обучение: {len(train)}. Тест: {len(test)}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model, attn_implementation="eager")
    encoder.requires_grad_(False)
    last_block = encoder.transformer.layer[-1]
    last_block.requires_grad_(True)
    encoder.eval()
    head = torch.nn.Linear(encoder.config.dim, len(labels))

    def encode_prefix(rows):
        texts, targets = zip(*rows)
        tokens = tokenizer(list(texts), padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
        # Первые пять блоков заморожены. Считаем их выход один раз.
        with torch.no_grad():
            hidden = encoder.embeddings(tokens.input_ids)
            for block in encoder.transformer.layer[:-1]:
                hidden = block(hidden, attn_mask=tokens.attention_mask)[-1]
        return hidden, tokens.attention_mask, torch.tensor(targets)

    train_hidden, train_mask, train_y = encode_prefix(train)
    test_hidden, test_mask, test_y = encode_prefix(test)

    def predict_logits(hidden, mask):
        encoded = last_block(hidden, attn_mask=mask)[-1]
        return head(encoded[:, 0])

    def evaluate(stage):
        last_block.eval()
        with torch.no_grad():
            predictions = predict_logits(test_hidden, test_mask).argmax(dim=1)
        correct = int((predictions == test_y).sum())
        print(f"{stage}: {correct}/{len(test)} ({correct / len(test):.1%})")
        return predictions.tolist()

    evaluate("До обучения (случайно инициализированный классификатор)")
    optimizer = torch.optim.AdamW([
        {"params": last_block.parameters(), "lr": 1e-4},
        {"params": head.parameters(), "lr": 1e-3},
    ])
    last_block.train()
    for epoch in range(12):
        total_loss = 0.0
        for indices in torch.randperm(len(train)).split(6):
            optimizer.zero_grad()
            logits = predict_logits(train_hidden[indices], train_mask[indices])
            loss = torch.nn.functional.cross_entropy(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"Эпоха {epoch + 1:02d}: train loss = {total_loss / len(train):.3f}")

    predictions = evaluate("После обучения")
    for (message, target), prediction in zip(test, predictions):
        print(f"{message} | верно: {labels[target]} | модель: {labels[prediction]}")

    if args.save_dir:
        encoder.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        torch.save({"head": head.state_dict(), "labels": labels},
                   f"{args.save_dir}/classifier.pt")


if __name__ == "__main__":
    main()
