from fastapi import APIRouter, UploadFile
from asyncio import sleep as sleep_async
from uvicorn import run as run_server
from fastapi.responses import HTMLResponse
from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB, imencode, CAP_DSHOW
from pickle import load as pickle_load
from time import time
from pydantic import BaseModel
from sqlite3 import connect as sqlite_connect
from random import SystemRandom
from string import ascii_uppercase, digits
from os.path import exists as path_exists
from os import mkdir

import mediapipe as mp
import numpy as np

DATABASE = sqlite_connect('database.sqlite')
DATABASE_CURSOR = DATABASE.cursor()

router = APIRouter(prefix="/engine")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle_load(open('model.p', 'rb'))
model = model_dict['model']

labels_dict = {0:'hi',1:'toi la',2:'hoang lan'} 

class GetTokenModel(BaseModel):
    username: str
    password: str

async def VerifyUserToken(token: str):
    if not DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE token = '{token}'").fetchone():
        return False
    else:
        return True

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
                    if OUTPUT == []:
                        OUTPUT.append(predicted_character)
                        print(predicted_character)
                    else:
                        if OUTPUT[-1] != predicted_character:
                            OUTPUT.append(predicted_character)
                            print(predicted_character)

        return {"data": OUTPUT}
    except Exception as e:
        return {"error": str(e)}

@router.post("/video")
async def UploadVideo(file: UploadFile = UploadFile,token: str = None):
    if not DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE token = '{token}'").fetchone():
        return {"error": "Invalid token."}
    else:
        TimeNow = time()
        with open(f"data/temp/{TimeNow}.mp4", "wb") as buffer:
            buffer.write(file.file.read())

    return await HandleVideo(f"data/temp/{TimeNow}.mp4")

def GenToken():
    return ''.join(SystemRandom().choice(ascii_uppercase + digits) for _ in range(32))

@router.post("/regentoken")
async def GetToken(data: GetTokenModel):
    token = GenToken()
    while DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE token = '{token}'").fetchone():
        token = GenToken()
    DATABASE_CURSOR.execute(f"UPDATE DATA_USER SET token = '{token}' WHERE username = '{data.username}' AND password = '{data.password}'")
    DATABASE.commit()
    
    DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE username = '{data.username}' AND password = '{data.password}'")
    result = DATABASE_CURSOR.fetchone()
    if result:
        return {"token": result[2]}
    else:
        return {"error": "Invalid username or password, you can go to /register to create a new account."}

@router.post("/register")
async def Register(username: str, password: str):
    try:
        DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE username = '{username}'")
        result = DATABASE_CURSOR.fetchone()
        if result:
            return {"error": "Username already exists."}
        else:
            token = GenToken()
            while DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE token = '{token}'").fetchone():
                token = GenToken()
            DATABASE_CURSOR.execute(f"INSERT INTO DATA_USER (username, password,token) VALUES ('{username}', '{password}', '{token}')")
            DATABASE.commit()
            return {"token": f"{token}"}
    except Exception as e:
        return {"error": str(e)}
    
@router.post("/upload")
async def UploadFile(file: UploadFile = UploadFile,token: str = None):
    if VerifyUserToken(token)==False:
        return {"error": "Invalid token."}
    else:
        USERNAME = DATABASE_CURSOR.execute(f"SELECT * FROM DATA_USER WHERE token = '{token}'").fetchone()[0]
        if not path_exists(f"data/user/{USERNAME}"):
            mkdir(f"data/user/{USERNAME}")

        with open(f"data/user/{USERNAME}/{file.filename}", "wb") as buffer:
            buffer.write(file.file.read())
        return {"status": "success", "filename": file.filename, "username": USERNAME,"time_upload": time(),"size": file.file.seek(0,2)}
    
@router.post("/customtrain")
async def train(token:str):
    return {"status":"will be updated soon."}
    
    
    
