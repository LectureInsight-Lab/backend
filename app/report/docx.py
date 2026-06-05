"""DOCX 리포트 (python-docx).

편집 가능한 Word 문서로 출력. 강사/관리자 피드백 협업에 적합.
"""
from app.analysis.schemas import InstructorScorecard


def render(scorecard: InstructorScorecard, output_path: str) -> str:
    """Scorecard → DOCX 파일 경로 (placeholder)."""
    raise NotImplementedError
