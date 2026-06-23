"""리포트 API 라우트 (v2)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.analysis.schemas import InstructorScorecard
from app.core import store
from app.report import report_generator

router = APIRouter()

_MEDIA = {
    "html": "text/html",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class GenerateRequest(BaseModel):
    """``scorecard_id`` (instructor__date) 로 저장본을 쓰거나, ``scorecard`` 를 직접 전달."""

    scorecard_id: str | None = None
    scorecard: InstructorScorecard | None = None
    formats: list[str] = ["html", "docx"]


def _file_id(instructor_id: str, lecture_date: str, fmt: str) -> str:
    return f"{instructor_id}__{lecture_date}__{fmt}"


@router.post("/generate")
async def generate_report(req: GenerateRequest) -> dict:
    """InstructorScorecard → HTML/DOCX 리포트 생성.

    응답: {"files": {fmt: path}, "download": {fmt: "/api/v1/report/download/{file_id}"}}
    """
    card = req.scorecard
    if card is None and req.scorecard_id:
        card = store.get(req.scorecard_id)
    if card is None:
        raise HTTPException(status_code=404, detail="scorecard 또는 유효한 scorecard_id 가 필요합니다.")

    files = report_generator.generate(card, formats=req.formats)
    download = {
        fmt: f"/api/v1/report/download/{_file_id(card.instructor_id, card.lecture_date, fmt)}"
        for fmt in files
    }
    return {"files": files, "download": download}


@router.get("/download/{file_id}")
async def download_report(file_id: str):
    """생성된 리포트 다운로드. file_id = instructor__date__fmt."""
    try:
        instructor_id, lecture_date, fmt = file_id.split("__")
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 file_id 형식 (instructor__date__fmt)")

    path = report_generator.OUTPUT_ROOT / instructor_id / f"{lecture_date}.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"리포트 파일 없음: {path.name}")
    return FileResponse(
        path,
        media_type=_MEDIA.get(fmt, "application/octet-stream"),
        filename=path.name,
    )
