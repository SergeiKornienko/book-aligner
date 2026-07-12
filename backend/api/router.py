"""
Main API router that includes all endpoint routers.
"""

from fastapi import APIRouter
from backend.api.endpoints import upload, alignment, paragraphs, process

router = APIRouter()

router.include_router(upload.router, tags=["upload"])
router.include_router(alignment.router, tags=["alignment"])
router.include_router(paragraphs.router, tags=["paragraphs"])
router.include_router(process.router, tags=["process"])