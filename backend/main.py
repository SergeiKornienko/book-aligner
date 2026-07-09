from fastapi import FastAPI
from backend.routes import router

app = FastAPI(title="BookAligner", version="0.1.0")
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "BookAligner API is running"}