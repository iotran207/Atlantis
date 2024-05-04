from fastapi import APIRouter, UploadFile
from asyncio import sleep as sleep_async
from uvicorn import run as run_server
from fastapi.responses import HTMLResponse
from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB, imencode, CAP_DSHOW
from pickle import load as pickle_load
from time import time

import mediapipe as mp
import numpy as np

router = APIRouter(prefix="/stream")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle_load(open('api/model.p', 'rb'))
model = model_dict['model']

labels_dict = {0:'hi',1:'toi la',2:'hoang lan'}

async def HandleVideo(VIDEO_PATH: str):
    video_capture = VideoCapture(VIDEO_PATH)
    predicted_character = ''
    OUTPUT = []
    CHECK_FRAME = 0
    try:
        while True:
            data_aux = []
            x_ = []
            y_ = []
            ret, frame = video_capture.read()
            if not ret:
                break

            H, W, _ = frame.shape

            frame_rgb = cvtColor(frame, COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                CHECK_FRAME+=1
                if CHECK_FRAME == 1:
                    CHECK_FRAME = 0
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[0].x
                            y = hand_landmarks.landmark[0].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[0].x
                            y = hand_landmarks.landmark[0].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    print(predicted_character)
                    if OUTPUT == []:
                        OUTPUT.append(predicted_character)
                    else:
                        if OUTPUT[-1] != predicted_character:
                            OUTPUT.append(predicted_character)

        return {"data": OUTPUT}
    except Exception as e:
        return {"error": str(e)}

@router.post("/video")
async def UploadVideo(file: UploadFile):
    TimeNow = time()
    with open(f"api/data/{TimeNow}.mp4", "wb") as buffer:
        buffer.write(file.file.read())

    return await HandleVideo(f"api/data/{TimeNow}.mp4")

    

