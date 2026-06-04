"""ReportLab 기반 PDF 리포트 생성."""
from app.analysis.schemas import LectureAnalysis


def render(analysis: LectureAnalysis, output_path: str) -> str:
    """분석 결과를 PDF 파일로 출력 (placeholder)."""
    raise NotImplementedError
