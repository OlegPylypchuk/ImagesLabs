import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import clear_border
from scipy.ndimage import rotate

#Завантаження моделі з ЛР №1
model = tf.keras.models.load_model('mnist_model.keras')
print("Модель успішно завантажено!")

#Функція підготовки
def prepare_image(image_path):
    # 1. Завантажуємо в градаціях сірого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Помилка: не вдалося завантажити {image_path}")
        return np.zeros((28, 28))

    #Замилюємо картинку, що б прибрати лінії клітинок
    #але залишить жирну ручку.
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # THRESH_BINARY_INV одразу робить фон чорним, а ручку білою.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    binary = clear_border(binary)

    # Шукаємо контури цифри
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((28, 28))

    # Знаходимо найбільший контур
    biggest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest_contour)

    # Захист: якщо знайдений об'єкт менший за 2 пікселі (випадкова крапка), ігноруємо
    if w < 2 or h < 2:
        return np.zeros((28, 28))

    #Вирізаємо цифру
    padding = 3
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(binary.shape[1], x + w + padding)
    y_end = min(binary.shape[0], y + h + padding)

    cropped_digit = binary[y_start:y_end, x_start:x_end]

    #Потовщення, щоб тонка ручка стала товстим маркером
    kernel = np.ones((3, 3), np.uint8)
    thick_digit = cv2.dilate(cropped_digit, kernel, iterations=1)

    #Масштабування та розміщення на ідеальному полотні 28x28
    pil_img = Image.fromarray(thick_digit)
    pil_img = pil_img.resize((20, 20), getattr(Image, 'Resampling', Image).LANCZOS)

    final_img = Image.new('L', (28, 28), color=0)
    final_img.paste(pil_img, (4, 4))

    # 9. Нормалізація та модель ROF
    final_array = np.array(final_img) / 255.0
    denoised_img = denoise_tv_chambolle(final_array, weight=0.15)

    return denoised_img


# 3. Серія розпізнавань з обертанням (Алгоритм Голосування)
def recognize_with_rotations(img_array, model):
    angles = range(-15, 16, 15)

    print("\nПочинаємо серію розпізнавань:")

    # Створюємо масив для накопичення ймовірностей для кожної з 10 цифр
    accumulated_probs = np.zeros(10)

    images_by_angle = {}
    best_angle_for_class = {i: 0 for i in range(10)}
    max_prob_for_class = {i: 0.0 for i in range(10)}

    for angle in angles:
        rotated = rotate(img_array, angle, reshape=False)
        rotated = np.clip(rotated, 0, 1)

        input_data = rotated.reshape(1, 28, 28)

        # Отримуємо всі 10 ймовірностей (масив)
        predictions = model.predict(input_data, verbose=0)[0]

        # Додаємо ці ймовірності до загального "голосування"
        accumulated_probs += predictions

        max_prob = np.max(predictions)
        predicted_digit = np.argmax(predictions)

        print(f"Кут {angle:3}°: розпізнано як {predicted_digit} (ймовірність {max_prob * 100:.1f}%)")

        images_by_angle[angle] = rotated

        # Запам'ятовуємо, при якому куті кожна цифра виглядала найкраще
        for i in range(10):
            if predictions[i] > max_prob_for_class[i]:
                max_prob_for_class[i] = predictions[i]
                best_angle_for_class[i] = angle

    # Фінальне рішення
    # Перемагає цифра, яка набрала найбільшу СУМУ балів за всі спроби
    final_digit = int(np.argmax(accumulated_probs))

    # Беремо максимальну впевненість, яку мала ця переможна цифра
    final_prob = max_prob_for_class[final_digit]

    # Беремо той кут, де ця переможна цифра виглядала найвпевненіше
    final_angle = best_angle_for_class[final_digit]
    final_rotated_img = images_by_angle[final_angle]

    return final_digit, final_prob, final_angle, final_rotated_img


dataset_folder = 'dataset'

if not os.path.exists(dataset_folder):
    print(f"ПОМИЛКА: Папку '{dataset_folder}' не знайдено в поточному проєкті.")
else:
    print(f"=== Починаємо обробку датасету з папки: {dataset_folder} ===\n")

    total_images = 0
    visual_gallery = []

    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            image_path = os.path.join(dataset_folder, filename)

            try:
                prepared_img = prepare_image(image_path)
                digit, prob, angle, final_img = recognize_with_rotations(prepared_img, model)

                print(f"Файл: {filename:15} | Розпізнано: {digit} | Впевненість: {prob * 100:.1f}% | Кут: {angle}°")

                if len(visual_gallery) < 5:
                    visual_gallery.append((filename, final_img, digit, angle))

            except Exception as e:
                print(f"Х Помилка при обробці файлу {filename}: {e}\n")

    print(f"\n=== Обробку завершено. Проаналізовано файлів: {total_images} ===")

    #Виведення наочних результатів на екран
    if visual_gallery:
        print("\nВідкриваю вікно з наочними результатами (галерея)...")

        n = len(visual_gallery)
        plt.figure(figsize=(3 * n, 4))

        for i, (fname, img, dig, ang) in enumerate(visual_gallery):
            plt.subplot(1, n, i + 1)
            plt.imshow(img, cmap='gray')

            plt.title(f"{fname}\nМережа: {dig}\n(Кут {ang}°)", color='blue', fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()