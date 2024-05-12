from os import mkdir
from os.path import exists as path_exists
from pickle import load as pickle_load
from sqlite3 import connect as sqlite_connect
from time import time

import mediapipe as mp
import numpy as np
from api.utils import randomString
from cv2 import COLOR_BGR2RGB, VideoCapture, cvtColor
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

DATABASE = sqlite_connect('database.sqlite')
DATABASE_CURSOR = DATABASE.cursor()

router = APIRouter(prefix="/engine")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle_load(open('../model/model.p', 'rb'))
model = model_dict['model']

labels_dict = {0: 'hi', 1: 'toi la', 2: 'hoang lan'}


class User(BaseModel):
    username: str
    password: str


def verifyUserToken(token: str) -> bool:
    return not not DATABASE_CURSOR.execute(
        "SELECT * FROM DATA_USER WHERE token = ?", (token, )).fetchone()


async def handleVideo(video_path: str) -> dict[str, list[str] | str]:
    video_capture = VideoCapture(video_path)
    predicted_character = ''
    output: list[str] = []
    detected = True

    try:
        while True:
            data_aux: list[int] = []
            xs: list[int] = []
            ys: list[int] = []
            ret, frame = video_capture.read()
            if not ret:
                break

            H, W, _ = frame.shape

            frame_rgb = cvtColor(frame, COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                detected = True

            if not detected:
                continue

            detected = False
            for hand_landmarks in results.multi_hand_landmarks:
                for _ in range(len(hand_landmarks.landmark)):
                    xs.append(hand_landmarks.landmark[0].x)
                    ys.append(hand_landmarks.landmark[0].y)

                for _ in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[0].x - min(xs))
                    data_aux.append(hand_landmarks.landmark[0].y - min(ys))

            # unused variables
            # x1 = int(min(x_) * W) - 10
            # y1 = int(min(y_) * H) - 10
            # x2 = int(max(x_) * W) - 10
            # y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            if output == []:
                output.append(predicted_character)
                print(predicted_character)
                continue
            if output[-1] != predicted_character:
                output.append(predicted_character)
                print(predicted_character)

        return {"data": output}
    except Exception as e:
        return {"error": str(e)}


@router.post("/video")
async def uploadVideo(file: UploadFile | None = None,
                      token: str | None = None):
    if file is None:
        return {"error": "No file provided."}
    if token is None:
        return {"error": "No token provided."}

    if not DATABASE_CURSOR.execute("SELECT * FROM DATA_USER WHERE token = ?",
                                   (token, )).fetchone():
        return {"error": "Invalid token."}

    timestamp = time()
    with open(f"data/temp/{timestamp}.mp4", "wb") as buffer:
        buffer.write(file.file.read())

    return await handleVideo(f"data/temp/{timestamp}.mp4")


@router.post("/regentoken")
async def getToken(user: User):
    token = randomString(32)

    while DATABASE_CURSOR.execute("SELECT * FROM DATA_USER WHERE token = ?",
                                  (token, )).fetchone():
        token = randomString(32)

    DATABASE_CURSOR.execute(
        "UPDATE DATA_USER SET token = ? WHERE username = ? AND password = ?",
        (token, user.username, user.password))
    DATABASE.commit()

    DATABASE_CURSOR.execute(
        "SELECT * FROM DATA_USER WHERE username = ? AND password = ?",
        (user.username, user.password))

    result = DATABASE_CURSOR.fetchone()
    if not result:
        return {
            "error":
            "Invalid username or password, you can go to /register to create a new account."
        }

    return {"token": result[2]}


@router.post("/register")
async def register(username: str, password: str):
    try:
        DATABASE_CURSOR.execute("SELECT * FROM DATA_USER WHERE username = ?",
                                (username, ))
        if DATABASE_CURSOR.fetchone():  # If the username already exists
            return {"error": "Username already exists."}

        token = randomString(32)
        while DATABASE_CURSOR.execute(
                "SELECT * FROM DATA_USER WHERE token = ?",
            (token, )).fetchone():
            # If the token already exists
            token = randomString(32)

        DATABASE_CURSOR.execute(
            "INSERT INTO DATA_USER (username, password, token) VALUES (?, ?, ?)",
            (username, password, token))

        DATABASE.commit()

        return {"token": token}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/upload")
async def uploadFile(file: UploadFile | None = None, token: str | None = None):
    if file is None:
        return {"error": "No file provided."}
    if token is None:
        return {"error": "No token provided."}

    if verifyUserToken(token) == False:
        return {"error": "Invalid token."}

    username = DATABASE_CURSOR.execute(
        "SELECT * FROM DATA_USER WHERE token = ?", (token, )).fetchone()[0]

    if not path_exists(f"data/user/{username}"):
        mkdir(f"data/user/{username}")

    with open(f"data/user/{username}/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())

    return {
        "status": "success",
        "filename": file.filename,
        "username": username,
        "time_upload": time(),
        "size": file.file.seek(0, 2)
    }


@router.post("/customtrain")
async def train(token: str):
    return {"status": "will be updated soon."}
