import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_panorama(img_left_path, img_right_path):
    # 1. Завантаження зображень
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    if img_left is None or img_right is None:
        print(
            f"Помилка: Не вдалося завантажити фото. Перевір чи існують файли '{img_left_path}' та '{img_right_path}'.")
        return

    # Конвертуємо кольори для Matplotlib
    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

    # Переводимо в градації сірого для SIFT
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    #Знаходження ознак (SIFT)
    print("Шукаємо ключові точки...")
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

    #Співставлення ознак
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Знайдено надійних спільних точок: {len(good_matches)}")

    img_matches = cv2.drawMatches(img_left_rgb, keypoints_left, img_right_rgb, keypoints_right,
                                  good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 4. Трансформація (RANSAC) та Зшивання
    if len(good_matches) > 4:
        #Шукаємо матрицю від ПРАВОГО фото до ЛІВОГО
        # m.trainIdx - це точки правого фото, m.queryIdx - лівого
        src_pts = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        width_pano = w_left + w_right
        height_pano = max(h_left, h_right) * 2  # Беремо з запасом по висоті, якщо є нахил

        # Деформуємо ПРАВЕ фото, натягуючи його на координати лівого
        print("Склеюємо панораму...")
        pano = cv2.warpPerspective(img_right_rgb, M, (width_pano, height_pano))

        # ЛІВЕ фото вставляємо на початок координат
        pano[0:h_left, 0:w_left] = img_left_rgb

        #Візуалізація
        print("Готово! Відкриваю графіки...")
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        plt.imshow(img_matches)
        plt.title(f'Співставлення ознак (Знайдено {len(good_matches)} точок)')
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(pano)
        plt.title('Готова панорама (Без обрізки країв)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print("ПОМИЛКА: Недостатньо спільних точок.")


create_panorama('left.jpg', 'right.jpg')