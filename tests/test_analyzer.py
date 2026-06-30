"""tests/test_analyzer.py — analyzer 비동기 경로 (Gemini 호출은 mock).

실제 API 키 없이 컨텍스트 선택·프롬프트 조립·JSON 파싱·18병렬·앙상블/스코어까지 검증.
google.genai 는 지연 import 라 미설치 환경에서도 동작한다.
"""
from __future__ import annotations

import asyncio
import json

from app.analysis import analyzer, pipeline
from app.analysis.schemas import LectureDocument, Utterance


def _patch_llm(monkeypatch, payload: dict):
    async def fake_generate(system: str, user: str) -> str:
        return json.dumps(payload)

    monkeypatch.setattr(analyzer, "_generate", fake_generate)
    monkeypatch.setattr(analyzer.settings, "llm_cache_enabled", False)


def _doc() -> LectureDocument:
    lines = [
        Utterance(timestamp=f"09:{i % 60:02d}:00", speaker_id="a",
                  text=f"오늘은 목표를 안내하고 한번 해보세요 정리하면 이렇습니다 {i}",
                  seconds_from_start=i * 5)
        for i in range(60)
    ]
    return LectureDocument(lecture_date="2026-02-02", instructor_id="i", all_lines=lines,
                           intro_lines=lines[:10], middle_lines=lines[10:50], outro_lines=lines[50:])


def test_full_pipeline_mocked(monkeypatch):
    _patch_llm(monkeypatch, {"score": 4, "evidence": "근거", "strengths": "좋음",
                             "improvements": "개선", "confidence": 0.6})
    raw_text = "\n".join(
        f"<{u.timestamp}> a: {u.text}" for u in _doc().all_lines
    )
    card = asyncio.run(pipeline.analyze_raw_text(raw_text, "2026-02-02", "inst_demo"))
    assert len(card.item_scores) == 18
    assert 1.0 <= card.overall_score <= 5.0
