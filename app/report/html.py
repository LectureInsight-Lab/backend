"""HTML 리포트 (Jinja2).

레이더/추이 차트를 base64 인라인으로 박아 단일 .html 파일로 출력.
"""
from app.analysis.schemas import InstructorScorecard


def render(scorecard: InstructorScorecard, output_path: str) -> str:
    """Scorecard → HTML 파일 경로 (placeholder)."""
    raise NotImplementedError
