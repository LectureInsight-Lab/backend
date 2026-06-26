"""app/analysis/rubric_llm.py — 항목별 루브릭 LLM 채점 공통 헬퍼 (담당: 심소민).

`prompts/items/<stem>.yaml`(system + user_template) 로드 + Gemini(JSON) 호출.
google.generativeai / yaml / settings 는 **지연 import** — 키·패키지 없는 환경에서도
각 항목 모듈의 순수 채점 함수(`_score_*`) 단위테스트가 import 단계에서 깨지지 않도록 한다.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "items"
_prompt_cache: dict[str, dict] = {}


def load_prompt(stem: str) -> dict:
    """prompts/items/<stem>.yaml 로드 (캐시). 각 모듈은 자기 stem과 동일한 YAML을 쓴다."""
    if stem not in _prompt_cache:
        import yaml  # 지연 import

        with open(_PROMPTS_DIR / f"{stem}.yaml", encoding="utf-8") as f:
            _prompt_cache[stem] = yaml.safe_load(f) or {}
    return _prompt_cache[stem]


def render_user(prompt: dict, chunk: str) -> str:
    """user_template의 {chunk} 자리에 청크 텍스트를 치환 (str.format 대신 replace — 중괄호 안전)."""
    return prompt.get("user_template", "{chunk}").replace("{chunk}", chunk)


def _parse_json(raw: str) -> dict:
    """LLM 응답 문자열 → dict (코드펜스/잡텍스트 방어)."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)


_model = None


def _get_model():
    global _model
    if _model is None:
        import google.generativeai as genai  # 지연 import
        from app.core.config import settings

        genai.configure(api_key=settings.api_key)
        _model = genai.GenerativeModel(settings.llm_model)
    return _model


async def judge(system: str, user: str, sem: asyncio.Semaphore) -> dict:
    """Gemini 호출 → JSON dict. 실패 시 `{"_error": ...}`.

    ⚠️ Gemini thinking-token이 max_output_tokens를 잠식하므로 출력 토큰을 제한하지 않는다
    (memory: gemini-thinking-token-gotcha).
    """
    import google.generativeai as genai  # 지연 import
    from app.core.config import settings

    model = _get_model()
    cfg = genai.GenerationConfig(
        temperature=settings.llm_temperature,
        response_mime_type="application/json",
    )
    contents = f"{system}\n\n{user}"
    async with sem:
        try:
            resp = await asyncio.to_thread(model.generate_content, contents, generation_config=cfg)
            return _parse_json(resp.text)
        except Exception as e:  # noqa: BLE001
            return {"_error": str(e)}


def build(final_score, reason: str, evidence_items: list | None, mode: str) -> dict:
    """기존 score 모듈과 통일된 반환 형식."""
    return {
        "evidence": evidence_items if mode == "debug" else None,
        "reason": reason,
        "final_score": final_score,
    }
