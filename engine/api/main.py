from os import listdir
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']

for file in listdir('./routers'):
    if file.endswith('.py') and file != '__init__.py':
        module = file.replace('.py', '')
        exec(f'from routers.{module} import router')
        exec(f'app.include_router(router)')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
