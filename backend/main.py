from fastapi import FastAPI
from backend.api.router import router
from backend.logger import logger

app = FastAPI(title="BookAligner", version="0.1.0")
app.include_router(router)

logger.info("BookAligner API started")


@app.get("/")
async def root():
    return {"message": "BookAligner API is running"}