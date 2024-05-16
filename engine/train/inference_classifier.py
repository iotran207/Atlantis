import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image,ImageDraw,ImageFont

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

OUTPUT = []

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0:'xin chào',1:'giám kháo',2:'tôi là',3:'Bảo Anh',4:'đây là',5:'Dự án mã nguồn mở Atlantis engine'}
CHECK_FRAME = 0
predicted_character = ''

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:

        CHECK_FRAME+=1

        if CHECK_FRAME == 1:
            CHECK_FRAME = 0

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                if hand_landmarks == results.multi_hand_landmarks[0]:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                else:
                    pass # đang cập nhật thêm cho nhiều tay

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype("Roboto-Regular.ttf", 32) 
            text_width = font.getlength(predicted_character)
            text_height = font.getbbox(predicted_character)[3] - font.getbbox(predicted_character)[1]
            text_x = x1  
            text_y = y1 - 10 - text_height
            draw.text((text_x, text_y), predicted_character, font=font, fill=(0, 0, 0))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if predicted_character != '':
        if len(OUTPUT) == 0 or OUTPUT[-1] != predicted_character:
            OUTPUT.append(predicted_character)


    cv2.imshow('frame', frame)
    if cv2.waitKey(25)==ord('q'):
        break

print(OUTPUT)

cap.release()
cv2.destroyAllWindows()
