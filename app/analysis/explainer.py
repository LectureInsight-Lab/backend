"""스코어카드 항목별 자연어 해설(강점/개선점) 생성.

규칙엔진 지표(예: ``level_ok=100%``, ``Q-A 페어 없음``)와 LLM 근거를 사람이 읽기
쉬운 한국어로 풀어, 각 ``ItemScore`` 의 ``strengths`` / ``improvements`` 를 채운다.

표현 레이어만 담당한다 — 점수(structured)는 건드리지 않으므로 감사/재현성은 유지되고,
프론트 '해설' 칸과 DOCX 리포트가 동일한 자연어 출력을 그대로 소비한다.
Gemini 1회 호출(+ analyzer 캐싱 재사용). 실패해도 원본 텍스트를 유지한다.
"""
from __future__ import annotations

import json

from loguru import logger

from app.analysis import analyzer
from app.analysis.rubrics import rubric_line
from app.analysis.schemas import InstructorScorecard

_SYSTEM = (
    "당신은 강의 코칭 전문가입니다. 각 평가 항목에 대해 아래 네 가지를 자연스러운 한국어로 "
    "작성합니다.\n"
    "- grounds: 항목 표 '해설' 둘째 줄에 들어갈 '근거' 한 줄(아주 간결, 40자 내외). "
    "raw_basis(판정 근거/지표)의 핵심만 압축하되, 수치가 있으면 유지하세요. raw_basis 가 길어도 "
    "반드시 한 줄로 요약합니다.\n"
    "- reason: 첫 줄 '핵심 한 줄 코멘트'(딱 1문장, 간결하게). 점수가 높으면 잘하고 있는 점을, "
    "낮으면 개선 방향을 한 문장으로. 고득점에 억지 개선 지시를 넣지 말 것. grounds(둘째 줄)와 "
    "중복되지 않게 수치 나열 대신 의미 중심으로 쓰세요.\n"
    "- strengths: 이 항목에서 잘하고 있는 점 1문장.\n"
    "- improvements: 구체적이고 실행 가능한 개선 제안 1~2문장.\n"
    "각 항목의 'rubric'(채점 기준)에 근거해 작성하고, 지표 기호(예: level_ok=100%, "
    "relevance_high=100%, n=2)는 그대로 쓰지 말고 의미를 풀어 설명하세요.\n"
    "📌 한국어로만 쓰고 항목명을 영어로 병기하지 마세요. rubric 에 '주의'가 있으면 반드시 따르세요. "
    "특히 '언어 일관성'은 존댓말/반말 통일(어조·전문성)만 뜻하므로 '이해를 돕는다/방해한다'로 쓰지 "
    "말고, '발화 완결성'도 학습 이해와 인과로 엮지 마세요.\n"
    "⚠️ 어조: 분석은 추정이므로 단정적·과도한 표현('방해가 됩니다', '심각한')은 피하고 "
    "'~될 여지가 있습니다', '~할 수 있습니다', '~로 보입니다'처럼 완곡하고 신중하게 쓰세요."
)


def _build_user(card: InstructorScorecard) -> str:
    items = [
        {
            "item_id": it.item_id,
            "name": it.name,
            "score": it.final_score,
            "rubric": rubric_line(it.item_id),  # 채점 기준(세부기준+고/저득점 의미)
            "raw_basis": it.improvements,  # 판정 근거(규칙=지표/LLM=CoT, 상류에서 길이 제한됨)
        }
        for it in card.item_scores
    ]
    return (
        "다음은 한 강의의 항목별 평가 결과입니다. 각 항목에 대해 grounds(한 줄 근거), "
        "reason(첫 줄 코멘트), strengths(강점), improvements(개선 제안)를 작성하세요. "
        "⚠️ 입력에 있는 모든 item_id 를 하나도 빠짐없이 결과에 포함해야 합니다. "
        "아래 JSON 형식으로만 응답하세요:\n"
        '{"items": [{"item_id": 4, "grounds": "...", "reason": "...", "strengths": "...", '
        '"improvements": "..."}, ...]}\n\n'
        f"평가 결과:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


async def attach_explanations(card: InstructorScorecard) -> InstructorScorecard:
    """``card.item_scores`` 의 strengths/improvements 를 자연어로 덮어쓴다 (in-place)."""
    if not card.item_scores:
        return card

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_user(card)},
    ]
    logger.info(f"[explainer] 자연어 해설 생성 — {len(card.item_scores)}개 항목 (Gemini 1회)")
    try:
        raw = await analyzer._call_llm(messages)
        data = json.loads(raw)
        by_id = {int(d["item_id"]): d for d in data.get("items", [])}
    except Exception as e:  # noqa: BLE001 — 해설 실패가 분석 전체를 막지 않도록
        logger.warning(f"[explainer] 해설 생성 실패 — 원본 텍스트 유지: {type(e).__name__}: {e}")
        return card

    for it in card.item_scores:
        d = by_id.get(it.item_id)
        if not d:
            continue
        if d.get("grounds"):
            it.evidence = str(d["grounds"]).strip()   # 근거 줄(둘째 줄) — 항상 한 줄로 압축
        if d.get("reason"):
            it.reason = str(d["reason"]).strip()
        if d.get("strengths"):
            it.strengths = str(d["strengths"]).strip()
        if d.get("improvements"):
            it.improvements = str(d["improvements"]).strip()

    logger.success(f"[explainer] ✓ 해설 생성 완료 — {len(by_id)}개 항목 반영")
    return card
