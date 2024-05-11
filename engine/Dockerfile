FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./api /code/api

COPY ./train/model.p /code/api/model.p

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8345"]