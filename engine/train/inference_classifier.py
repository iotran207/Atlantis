import pickle

import cv2
import mediapipe as mp
import numpy as np

with open('../model/model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

output: list[str] = []

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'ok', 1: 'xin chao', 2: 'tam biet'}
detected = False
predicted_character = ''

while True:
    data_aux = []
    xs = []
    ys = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        detected = True

    if not detected:
        continue

    detected = False

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    for hand_landmarks in results.multi_hand_landmarks:
        for _ in range(len(hand_landmarks.landmark)):
            xs.append(hand_landmarks.landmark[0].x)
            ys.append(hand_landmarks.landmark[0].y)

        for _ in range(len(hand_landmarks.landmark)):
            data_aux.append(hand_landmarks.landmark[0].x - min(xs))
            data_aux.append(hand_landmarks.landmark[0].y - min(ys))

    x1 = int(min(xs) * W) - 10
    y1 = int(min(ys) * H) - 10

    x2 = int(max(xs) * W) - 10
    y2 = int(max(ys) * H) - 10

    prediction = model.predict([np.asarray(data_aux)])

    print(prediction)

    predicted_character = labels_dict[int(prediction[0])]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    if predicted_character != '':
        if len(output) == 0 or output[-1] != predicted_character:
            output.append(predicted_character)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

print(output)

cap.release()
cv2.destroyAllWindows()
