"""LLM 게이트웨이 (Gemini) — 캐싱 · 재시도 포함.

explainer(항목 해설)·narrative(종합 분석)가 공유하는 단일 LLM 호출 함수 ``_call_llm``.
동일 (model, temperature, system, user) → ``.cache/llm`` 에 캐싱하고,
tenacity 로 일시적 오류를 재시도한다.

NOTE: 옛 v2 18항목 분석 경로(analyze_item / analyze_lecture + behavior_tagger / ensemble)는
제거되었다. 실제 항목 채점은 ``app/analysis/pipeline.py`` 와 항목별 모듈이 담당한다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

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
    """[{role: system}, {role: user}] → LLM 응답 문자열 (캐시 우선)."""
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
