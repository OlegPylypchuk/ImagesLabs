import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Завантаження даних
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),

    # Перший згортковий блок (шукає прості лінії та кути)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Другий згортковий блок (збирає лінії у складні фігури)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Фінальний блок прийняття рішень
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Захист від перенавчання
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("--- Навчання моделі  ---")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Оцінка
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nТочність на тестових даних: {test_acc * 100:.2f}%')

# Зберігаємо модель
model.save('mnist_model.keras')

plt.figure(figsize=(12, 4))

# Графік точності
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Навчальна вибірка', marker='o')
plt.plot(history.history['val_accuracy'], label='Валідаційна вибірка', marker='o')
plt.title('Графік зміни точності')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.grid(True)

#Графік втрат
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Навчальна вибірка', marker='o', color='red')
plt.plot(history.history['val_loss'], label='Валідаційна вибірка', marker='o', color='orange')
plt.title('Графік зміни функції втрат')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()
plt.grid(True)

plt.show()

#Аналіз помилок
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

incorrect_indices = np.where(predicted_classes != y_test)[0]
print(f"\nЗнайдено {len(incorrect_indices)} помилок з 10000 тестових зображень.")

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    idx = incorrect_indices[i]
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Мережа: {predicted_classes[idx]}\nНасправді: {y_test[idx]}",
              color='red' if predicted_classes[idx] != y_test[idx] else 'black')
    plt.axis('off')

plt.tight_layout()
plt.show()