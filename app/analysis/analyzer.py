"""LLM 분석 (3단계) — Paper #2 기반 (Gemini).

18개 항목에 대해 비동기 병렬로 Gemini 를 호출하고 구조화 JSON 응답을 받는다.

각 호출은:
    1) ChecklistItem 의 context_strategy 로 컨텍스트 청크 결정
       - intro / middle / outro → 해당 구간 발화를 청크로
       - full_sample / keyword  → embedder.search(항목 쿼리, top_k)
    2) BehaviorProfile 에서 해당 항목 BoW 카운트 추출
    3) templates.build_messages 로 프롬프트 조립
    4) Gemini generate_content (response_mime_type=application/json) 호출
    5) LLMItemRaw 로 파싱

- temperature 는 settings.llm_temperature 고정 (Paper #6)
- tenacity 재시도 (일시적 오류)
- 동일 (model, temp, system, user) → .cache/llm 캐싱
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from app.analysis import embedder, templates
from app.analysis.schemas import (
    BehaviorProfile,
    IndexedChunk,
    ItemBoW,
    LectureDocument,
    LectureIndex,
    LLMItemRaw,
)
from app.core.checklist import Checklist, ChecklistItem, ContextStrategy
from app.core.config import settings

_SEGMENT_STRATEGIES = {
    ContextStrategy.INTRO: "intro_lines",
    ContextStrategy.MIDDLE: "middle_lines",
    ContextStrategy.OUTRO: "outro_lines",
}

# Gemini 클라이언트 싱글턴 (지연 초기화 — 키/패키지 없는 환경에서 import 실패 방지)
_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.api_key:
            raise RuntimeError("API_KEY 가 비어 있습니다. .env 에 Gemini API_KEY 를 설정하세요.")
        from google import genai

        _client = genai.Client(api_key=settings.api_key)
    return _client


# ── 컨텍스트 선택 ─────────────────────────────────────────────
def _segment_chunks(lines, chunk_lines: int, top_k: int) -> list[IndexedChunk]:
    """구간 발화(intro/middle/outro)를 청크로 분할해 최대 top_k 개 반환."""
    chunks: list[IndexedChunk] = []
    for chunk_id, start in enumerate(range(0, len(lines), chunk_lines)):
        window = lines[start : start + chunk_lines]
        if not window:
            continue
        chunks.append(
            IndexedChunk(
                chunk_id=chunk_id,
                start_timestamp=window[0].timestamp,
                end_timestamp=window[-1].timestamp,
                text=" ".join(u.text for u in window),
                embedding=None,
                line_indices=list(range(start, start + len(window))),
            )
        )
    return chunks[:top_k]


def _item_query(item: ChecklistItem) -> str:
    """RAG 검색 쿼리 = 항목명 + 평가 기준(criteria) 결합."""
    prompt = templates.load_item_prompt(item.prompt)
    criteria = prompt.get("criteria") or []
    return " ".join([item.name, *[str(c) for c in criteria]])


def _select_chunks(item: ChecklistItem, document: LectureDocument, index: LectureIndex) -> list[IndexedChunk]:
    seg_attr = _SEGMENT_STRATEGIES.get(item.context_strategy)
    if seg_attr is not None:
        lines = getattr(document, seg_attr)
        return _segment_chunks(lines, settings.rag_chunk_lines, settings.rag_top_k)
    # full_sample / keyword → 키워드 검색
    return embedder.search(index, _item_query(item), settings.rag_top_k)


# ── Gemini 호출 + 캐싱 ────────────────────────────────────────
def _cache_path(key: str) -> Path:
    return Path(settings.llm_cache_path) / f"{key}.json"


def _cache_key(system: str, user: str) -> str:
    payload = f"{settings.llm_model}|{settings.llm_temperature}|{system}|{user}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
async def _generate(system: str, user: str) -> str:
    """Gemini 호출 → JSON 문자열 (재시도)."""
    from google.genai import types

    client = _get_client()
    response = await client.aio.models.generate_content(
        model=settings.llm_model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=settings.llm_temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text


async def _call_llm(messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")

    cache_file = _cache_path(_cache_key(system, user))
    if settings.llm_cache_enabled and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    raw = await _generate(system, user)

    if settings.llm_cache_enabled:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(raw, encoding="utf-8")
    return raw


# ── 응답 파싱 ─────────────────────────────────────────────────
def _parse_response(raw: str, item: ChecklistItem, chunks: list[IndexedChunk]) -> LLMItemRaw:
    """JSON 문자열 → LLMItemRaw (필드 보정·클램프)."""
    text = raw.strip()
    if text.startswith("```"):                # 코드펜스 방어
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    data = json.loads(text)

    score = max(1.0, min(5.0, float(data.get("score", 3.0))))
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    return LLMItemRaw(
        item_id=item.id,
        score=score,
        evidence=str(data.get("evidence", "")),
        strengths=str(data.get("strengths", "")),
        improvements=str(data.get("improvements", "")),
        confidence=confidence,
        used_chunk_ids=[c.chunk_id for c in chunks],
    )


# ── 단일/전체 분석 ────────────────────────────────────────────
async def analyze_item(
    item: ChecklistItem,
    document: LectureDocument,
    index: LectureIndex,
    behavior: BehaviorProfile,
) -> LLMItemRaw:
    """단일 항목 LLM 분석."""
    chunks = _select_chunks(item, document, index)
    bow = behavior.items.get(item.id, ItemBoW())
    messages = templates.build_messages(item, chunks, bow, document=document, top_k=settings.rag_top_k)
    raw = await _call_llm(messages)
    return _parse_response(raw, item, chunks)


async def analyze_lecture(
    document: LectureDocument,
    index: LectureIndex,
    behavior: BehaviorProfile,
    checklist: Checklist,
) -> list[LLMItemRaw]:
    """18개 항목 병렬 분석."""
    results = await asyncio.gather(
        *[analyze_item(it, document, index, behavior) for it in checklist.items]
    )
    return list(results)
