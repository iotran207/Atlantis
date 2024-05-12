from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.engine import router as engine_router

app = FastAPI()

origins = ['*']

app.include_router(engine_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {"message": "Hello World!"}
