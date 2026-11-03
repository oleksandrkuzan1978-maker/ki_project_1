# Подключаем PyTorch и конвейер Stable Diffusion для генерации изображений по тексту.
# Импорт Image из Pillow ниже напрямую не используется.
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


# Выбираем видеокарту с CUDA, если она доступна; иначе используем процессор.
def setup_device():
    # Проверка MPS для устройств Apple закомментирована и сейчас не выполняется.
    # if torch.backends.mps.is_available():
    #     return "mps"
    if torch.cuda.is_available():
        return 'cuda'
    return "cpu"


# Загружаем модель SD-Turbo и переносим её на выбранное устройство.
def load_model(device):
    print("Loading Stable Diffusion model...")

    # Идентификатор готовой модели в Hugging Face.
    model_id = "stabilityai/sd-turbo"

    # Создаём конвейер; недостающие файлы модели загружаются, имеющиеся берутся из кэша.
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        # На CUDA используем float16 для экономии памяти, на CPU — float32.
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        # Передаём параметр возобновления загрузки; его поддержка зависит от версии библиотек.
        resume_download=True
    )

    # Переносим компоненты модели на CUDA или CPU.
    pipe = pipe.to(device)

    # Следующий блок настройки attention отключён и не влияет на выполнение.
    # ускорение (если доступно)
    # if device == "mps":
    #     pipe.enable_attention_slicing()

    print("Model loaded successfully!")
    return pipe


# Генерируем картинку по описанию, нежелательным признакам и числу шагов.
def generate_image(pipe, prompt, negative_prompt="", num_steps=25):
    """Генерация изображения"""
    print(f"\nGenerating: {prompt}")

    # Запускаем генерацию и берём первое изображение из результата.
    # num_inference_steps — число шагов генерации, а не количество картинок.
    # guidance_scale явно не задан: действует значение по умолчанию конвейера.
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps
    ).images[0]

    return image


# Сохраняем изображение в файл; формат определяется расширением имени.
def save_image(image, filename):
    image.save(filename)
    print(f"Saved: {filename}")


# Открываем изображение во внешнем просмотрщике, выбранном системой.
def show_image(image):
    image.show()


# Основной сценарий: выбираем устройство, загружаем модель и обрабатываем описания.
def main():
    device = setup_device()
    print(f"Using device: {device}")

    pipe = load_model(device)

    # Каждое описание в списке задаёт отдельное изображение.
    prompts = [
        """Portrait of a young woman, natural skin texture, 
        soft golden hour lighting, 85mm lens, 
        shallow depth of field, cinematic photography, 
        detailed eyes, realistic colors"""
    ]

    # Перечисляем нежелательные признаки для механизма guidance конвейера.
    negative_prompt = "blurry, bad quality, distorted, ugly"

    # По очереди генерируем картинки; i используется для нумерации файлов.
    for i, prompt in enumerate(prompts):
        # Обрабатываем каждое описание отдельно, чтобы ошибка не прервала весь цикл.
        try:
            image = generate_image(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                # Здесь задаём 4 шага вместо значения 25 по умолчанию в функции.
                num_steps=25
            )
            
            # Сохраняем PNG в текущей рабочей папке и открываем для просмотра.
            # При повторном запуске файл с тем же именем будет перезаписан.
            filename = f"generated_image_{i + 1}.png"
            save_image(image, filename)
            show_image(image)

        # Выводим ошибку генерации, сохранения или показа и переходим к следующему описанию.
        except Exception as e:
            print(f"Error generating image: {e}")


# Запускаем main только при прямом запуске этого файла.
if __name__ == "__main__":
    main()