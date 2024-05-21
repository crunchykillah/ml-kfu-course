import cv2
import mediapipe as mp
import face_recognition


# Инициализация Mediapipe для обнаружения рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Загрузка и обучение модели распознавания лиц
your_face_encodings = []
your_face_name = "Artemii Volkov"

# Загрузка изображений и получение их кодировок
image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg"]  # Фото для обучения модели
for image_path in image_paths:
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        encoding = encodings[0]
        your_face_encodings.append(encoding)

# Функция для определения количества поднятых пальцев
def count_fingers(hand_landmarks):
    count = 0
    tips_ids = [4, 8, 12, 16, 20]

    # Проверяем подняты ли пальцы
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:  # Большой палец
        count += 1
    for tip_id in tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:  # Остальные пальцы
            count += 1
    return count

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось захватить кадр.")
            continue

        frame = cv2.flip(frame, 1)

        # Преобразование изображения в формат RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обнаружение лиц на кадре
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Обнаружение рук
        results = hands.process(rgb_frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(your_face_encodings, face_encoding)
            name = "undefined"

            if True in matches:
                name = your_face_name

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Подсчет поднятых пальцев
                        fingers_count = count_fingers(hand_landmarks)

                        if fingers_count == 1:
                            name = "Artemii"
                        elif fingers_count == 2:
                            name = "Volkov"

            # Отображение имени или "undefined"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Отображение обработанного кадра
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
