"""차트 생성 (matplotlib / plotly).

레이더 차트: 카테고리별 점수 시각화
추이 차트: 날짜별 종합 점수 시계열

HTML 리포트에는 base64 이미지로 인라인 삽입.
DOCX 에는 png 파일로 임시 저장 후 삽입.
"""
from app.analysis.schemas import InstructorScorecard


def radar_chart(scorecard: InstructorScorecard) -> bytes:
    """카테고리별 점수 레이더 차트 → PNG bytes (placeholder)."""
    raise NotImplementedError


def trend_chart(scorecard: InstructorScorecard) -> bytes:
    """날짜별 종합 점수 추이 차트 → PNG bytes (placeholder)."""
    raise NotImplementedError


def to_base64(png_bytes: bytes) -> str:
    """PNG → base64 (HTML img src 용) (placeholder)."""
    raise NotImplementedError
