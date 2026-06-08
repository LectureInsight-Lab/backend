"""리포트 생성 통합 진입점 (6단계).

스코어카드 + 차트 → HTML / DOCX 동시 생성.

산출물 위치: ``outputs/{instructor_id}/{lecture_date}.{html,docx}``
"""
from app.analysis.schemas import InstructorScorecard
from app.report import charts, docx, html


def generate(scorecard: InstructorScorecard, formats: list[str] = ("html", "docx")) -> dict[str, str]:
    """선택된 포맷으로 리포트 생성 → {format: file_path} (placeholder)."""
    raise NotImplementedError
