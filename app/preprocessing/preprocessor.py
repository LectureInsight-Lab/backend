"""STT 파싱 + 구간 분리 (1단계).

STT 포맷: ``<HH:MM:SS> speaker_id: 발화 텍스트``

intro/middle/outro 3구간으로 분리:
    intro  = 시작 후 N분
    outro  = 종료 전 M분
    middle = 그 사이

구간 분리 기준은 ``configs/checklist.yaml`` 의 ``segments`` 섹션에서 로드.

TODO:
- 정규식 기반 라인 파싱
- 화자 분리 (강사 vs 학생) — 강사 ID 휴리스틱
- 보조 통계 산출 호출 (eda.py)
"""
from app.analysis.schemas import LectureDocument, Utterance


def parse(raw_text: str) -> list[Utterance]:
    """원본 STT를 Utterance 리스트로 파싱 (placeholder)."""
    raise NotImplementedError


def split_segments(
    utterances: list[Utterance],
    intro_minutes: int = 20,
    outro_minutes: int = 15,
) -> tuple[list[Utterance], list[Utterance], list[Utterance]]:
    """(intro, middle, outro) 3구간 분리 (placeholder)."""
    raise NotImplementedError


def build_document(
    raw_text: str,
    lecture_date: str,
    instructor_id: str,
    intro_minutes: int = 20,
    outro_minutes: int = 15,
) -> LectureDocument:
    """raw_text → LectureDocument 한 번에 (placeholder)."""
    raise NotImplementedError
