# Подключаем PyTorch, модель CLIP, обработку изображений и загрузку по HTTP.
import torch  
from transformers import CLIPModel, CLIPProcessor 
from PIL import Image  
import requests  
from io import BytesIO  

# Загружаем готовые веса CLIP и процессор для подготовки текста и изображений.
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Загружаем изображение по URL и открываем полученные байты через Pillow.
def load_image_from_url(url):
    response = requests.get(url)  
    return Image.open(BytesIO(response.content))  


# Открываем локальное изображение; в main эта функция сейчас не вызывается.
def load_image_from_file(file_path):
    return Image.open(file_path)  


# Сравниваем каждый текст с каждым изображением в общем пространстве признаков CLIP.
def text_to_image_similarity(model, processor, text_queries, images):
    # Превращаем тексты в тензоры PyTorch, дополняя последовательности до общей длины.
    text_inputs = processor(text=text_queries, return_tensors='pt', padding=True)
    # Отключаем расчёт градиентов: используем готовую модель без обучения.
    with torch.no_grad():  
        # Получаем вектор признаков для каждого текста.
        text_features = model.get_text_features(**text_inputs)  
        # Нормируем векторы до длины 1, чтобы сравнивать их направления.
        text_features /= text_features.norm(dim=-1, keepdim=True)  

    
    # Подготавливаем изображения: размер, нормализация пикселей и преобразование в тензоры.
    image_inputs = processor(images=images, return_tensors="pt")
    # Отключаем расчёт градиентов: используем готовую модель без обучения.
    with torch.no_grad():  
        # Получаем вектор признаков изображения.
        image_features = model.get_image_features(**image_inputs)  
        # Приводим векторы изображений к единичной длине.
        image_features /= image_features.norm(dim=-1, keepdim=True)  

    # Вектор этой CLIP-модели имеет 512 компонентов; число строк зависит от входных данных.
    # Следующий исходный комментарий с размерностью 1028 неточен.
    ### text_features -> shape(4, 1028) ; image_features -> shape(2, 1028)
    print(f'Text shape after processing: {text_features.shape}, Image shape after processing: {image_features.shape}')
    
    # Произведение нормированных векторов даёт косинусное сходство, а не вероятность.
    # Матрица результата: число текстов × число изображений; .T меняет оси местами.
    similarity = (text_features @ image_features.T) ### shape(4, 2)
    # Переносим результат на CPU и преобразуем его в массив NumPy.
    return similarity.cpu().numpy()  


# Основной сценарий: загружаем картинки и сравниваем одну из них с описаниями классов.
def main():
    # Переменная device ниже не используется; модель по умолчанию остаётся на CPU.
    device = 'cpu'
    print("\nExample 1: Comparing text prompts to online images")
    # Адреса двух изображений для сравнения.
    image_urls = [
        "https://cdn.shopify.com/s/files/1/0086/0795/7054/files/Golden-Retriever.jpg?v=1645179525",  
        "https://miro.medium.com/v2/resize:fit:1400/1*tMKkGydXuiOBOb15srANvg@2x.jpeg"  
    ]

    
    # Пробуем загрузить обе картинки; ошибка любой загрузки завершает main.
    try:
        images = [load_image_from_url(url) for url in image_urls]  
        print(f"Successfully loaded {len(images)} images")
    except Exception as e:
        print(f"Error loading images: {e}")  
        print("Falling back to local images if available...")
        
        # Несмотря на сообщение выше, переход к локальным файлам здесь не реализован.
        return

    
    # Отключённый пример: задаём тексты, вычисляем сходство и выводим оценки.
    # Умножение сходства на 100 не превращает его в вероятность.
    # text_queries = ["a dog, golden retriever", "a car", "a sunset with sea", "a person"]

    
    # similarities = text_to_image_similarity(model, processor, text_queries, images)

    
    # print("\nSimilarity Results (%):")
    # for i, text in enumerate(text_queries):
    #     print(f"\nText: '{text}'")
    #     for j, url in enumerate(image_urls):
    #         print(f"  Image {j + 1}: {similarities[i][j] * 100:.2f}%")  

    
    print("\nExample 2: Zero-shot image classification")

    
    # Задаём возможные классы словами: zero-shot не требует обучения на этих классах.
    labels = ["a photo of a dog", "a photo of a cat", "a photo of a car", "a photo of a sunset", "a photo of sand with a sea", "a photo of a beach with a blue sky"]

    
    # Для классификации выбираем второе загруженное изображение (индекс 1).
    image_input = processor(images=images[1], return_tensors="pt")

    
    # Подготавливаем текстовые описания всех возможных классов.
    text_inputs = processor(text=labels, return_tensors='pt', padding=True) 

    
    # Отключаем расчёт градиентов: используем готовую модель без обучения.
    with torch.no_grad():  
        # Получаем вектор признаков изображения.
        image_features = model.get_image_features(**image_input)  
        # Получаем вектор признаков для каждого текста.
        text_features = model.get_text_features(**text_inputs)  

        
        # Приводим векторы изображений к единичной длине.
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # Нормируем векторы до длины 1, чтобы сравнивать их направления.
        text_features /= text_features.norm(dim=-1, keepdim=True)

        
        # Умножаем сходства на 100 и применяем softmax: получаем относительные
        # оценки среди заданных классов, а не гарантию правильности классификации.
        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # Получаем массив оценок для единственного выбранного изображения.
        probs = logits_per_image.cpu().numpy()[0]  

    
    print("\nClassification Results:")
    # Печатаем долю каждого класса в процентах; вместе они дают примерно 100%.
    for i, label in enumerate(labels):
        print(f"{label}: {probs[i] * 100:.2f}%")  


# Запускаем main только при прямом запуске файла, а не при его импорте.
if __name__ == "__main__":
    main()  