from os import listdir
from fastapi import FastAPI

app = FastAPI()

for file in listdir('./api/routers'):
    if file.endswith('.py') and file != '__init__.py':
        module = file.replace('.py', '')
        exec(f'from api.routers.{module} import router')
        exec(f'app.include_router(router)')

@app.get("/")
async def home():
    return {"message": "Hello World!"}