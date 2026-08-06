"""
서버가 살아있는지 확인
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}