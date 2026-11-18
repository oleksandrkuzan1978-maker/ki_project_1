import torch  # Библиотека для работы с нейросетями; напрямую в этом коде не используется.
from transformers import pipeline  # Готовый инструмент для запуска моделей.
import time  # Нужен для измерения времени распознавания.

# Выполняем распознавание на процессоре.
device = 'cpu'

# Создаём обработчик распознавания речи с моделью Whisper Base.
# При первом запуске модель загружается из интернета.
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device
)

# Относительный путь отсчитывается от папки, из которой запущена программа.
audio_file = "Sting - Shape Of My Heart.mp3"
start = time.time()  # Запоминаем время перед обработкой аудио.
result = pipe(
    audio_file,
    return_timestamps=True  # Просим вернуть текст и временные границы сегментов.
)
end = time.time()  # Запоминаем время после обработки.

# Выводим длительность обработки в секундах с двумя знаками после точки.
print(f'Total time: {end-start:.2f}')
# print(result)  # Можно раскомментировать, чтобы увидеть весь результат модели.
print("Полная транскрипция:")
# Берём полный распознанный текст; end добавляет две строки переноса.
print(result["text"], end='\n\n')

print("\nСегменты:")
# Перебираем фрагменты; если ключа chunks нет, используем пустой список.
for chunk in result.get("chunks", []):
    start, end = chunk["timestamp"]  # Начало и конец фрагмента в секундах.
    text = chunk["text"]  # Текст этого фрагмента.
    # Выводим временные границы и текст фрагмента.
    print(f"[{start:.2f} - {end:.2f}] {text}", end='\n\n')
