"""강의 분석 종합 자연어 해설(overall_feedback) + 요약(summary) 생성.

요청 흐름:
  (1) 점수·항목 해설·루브릭을 근거로 '종합 분석'을 먼저 생성(overall_feedback)
  (2) 그 분석을 다시 '요약'(summary) — 상단 분석 요약 박스용
톤은 대상에 따라 다르다:
  - 단일 강의: "이번 강의" 시점
  - 입력 파일 종합(aggregate): "이 강사/전반적으로" 종합 시점 ("이번 강의" 금지)

단일/종합 모두 프론트가 호출하는 ``POST /analysis/narrative`` 가 이 모듈을 사용한다.
LLM 실패 시 점수 기반 폴백 문장을 반환해 화면이 비지 않게 한다.
"""
from __future__ import annotations

import json

from loguru import logger

from app.analysis import analyzer
from app.analysis.rubrics import rubric_line
from app.analysis.schemas import InstructorScorecard

# 공통 어조 지침 — 분석은 추정이므로 단정형 대신 완곡·신중한 표현 사용.
_TONE = (
    "⚠️ 어조: 분석 결과는 추정이므로 단정적·과도하게 강한 표현('방해가 됩니다', '심각한', "
    "'전혀 ~못합니다')은 피하고, '방해가 될 여지가 있습니다', '~할 수 있습니다', '~로 보입니다', "
    "'~인 경향이 있습니다'처럼 완곡하고 신중한 어조로 서술하세요."
)

_FEEDBACK_SYSTEM_SINGLE = (
    "당신은 강의 코칭 전문가입니다. 한 강사의 '단일 강의' 분석 결과를 바탕으로 '이번 강의'에 "
    "대한 '상세한 종합 분석'을 작성합니다. 다음을 모두 담아 2~3개 단락, 8문장 이상으로 충실하게 "
    "쓰세요:\n"
    "(1) 강의 전반에 대한 종합 평가,\n"
    "(2) 잘한 점을 카테고리·항목과 함께 구체적으로,\n"
    "(3) 약점과 가장 효과적인 개선 방향을 구체적·실행 가능하게.\n"
    "점수를 단순 나열하지 말고 흐름 있게 코칭하듯 서술하세요. 각 항목의 rubric(채점 기준)에 "
    "근거하고, 단락은 빈 줄로 구분하세요.\n" + _TONE
)
_FEEDBACK_SYSTEM_AGG = (
    "당신은 강의 코칭 전문가입니다. 한 강사의 '여러 강의를 종합한' 분석 결과를 바탕으로 이 "
    "강사의 전반적인 강의 경향에 대한 '상세한 종합 분석'을 작성합니다. 다음을 모두 담아 "
    "2~3개 단락, 8문장 이상으로 충실하게 쓰세요:\n"
    "(1) 전반적 평가와 점수 추이(향상/유지/하락)의 의미,\n"
    "(2) 여러 강의에 걸쳐 꾸준히 나타나는 강점을 카테고리·항목과 함께 구체적으로,\n"
    "(3) 반복되는 약점과 가장 효과적인 우선 개선 방향을 구체적·실행 가능하게.\n"
    "⚠️ '이번 강의' 같은 단일 강의 표현은 쓰지 말고 '이 강사는/전반적으로/여러 강의에 걸쳐'처럼 "
    "종합적 시점으로 서술하세요. 각 항목의 rubric(채점 기준)에 근거하고, 단락은 빈 줄로 구분하세요.\n"
    + _TONE
)
_SUMMARY_SYSTEM = (
    "당신은 편집자입니다. 주어진 강의 종합 해설을 핵심만 2~3문장으로 요약합니다. 새로운 정보를 "
    "만들지 말고 원문 해설의 요지(종합 평가·핵심 강점·우선 개선점)만 압축하세요. 원문과 같은 "
    "시점(단일 강의면 '이번 강의', 종합이면 종합 시점)을 유지하세요."
)


def _card_context(card: InstructorScorecard, is_aggregate: bool, lecture_count: int) -> str:
    cats = "\n".join(f"- {c.category}: {c.score}점 (가중치 {c.weight})" for c in card.category_scores)
    items = "\n".join(
        f"- [{it.item_id}] {it.name}: {it.final_score}점"
        + (f" | 채점기준 {rubric_line(it.item_id)}" if rubric_line(it.item_id) else "")
        + (f" | 해설: {it.reason}" if it.reason else "")
        for it in sorted(card.item_scores, key=lambda x: x.item_id)
    )
    head = f"강사: {card.instructor_id}\n종합 점수: {card.overall_score}/5.0\n"
    if is_aggregate:
        head += f"분석 대상: {lecture_count}개 강의 종합 ({card.lecture_date})\n"
        if card.trend_label:
            head += f"점수 추이: {card.trend_label} (기울기 {card.trend_slope})\n"
    else:
        head += f"강의 일자: {card.lecture_date}\n"
    return f"{head}\n[카테고리 점수]\n{cats}\n\n[항목별 점수·채점기준·해설]\n{items}"


def _fallback(card: InstructorScorecard, is_aggregate: bool) -> dict:
    subject = "이 강사의 강의들은" if is_aggregate else "이번 강의는"
    return {
        "overall_feedback": (
            f"{subject} 종합 {card.overall_score}/5.0 수준입니다. "
            "자세한 종합 해설 생성에 실패해 점수 기반 요약만 제공합니다."
        ),
        "summary": f"종합 {card.overall_score}/5.0.",
    }


async def _gen_json(system: str, user: str, key: str) -> str | None:
    try:
        raw = await analyzer._call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return str(json.loads(raw).get(key, "")).strip() or None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[narrative] '{key}' 생성 실패: {type(e).__name__}: {e}")
        return None


async def generate_narrative(
    card: InstructorScorecard,
    is_aggregate: bool = False,
    lecture_count: int = 1,
) -> dict:
    """{overall_feedback, summary} 반환. (1) 종합 해설 → (2) 그 해설을 요약."""
    ctx = _card_context(card, is_aggregate, lecture_count)
    mode = "종합" if is_aggregate else "단일"
    logger.info(f"[narrative] ▶ 종합 분석 생성 ({mode}) — instructor={card.instructor_id}")

    # (1) 종합 분석
    sys_fb = _FEEDBACK_SYSTEM_AGG if is_aggregate else _FEEDBACK_SYSTEM_SINGLE
    overall = await _gen_json(
        sys_fb,
        "다음 분석 결과로 종합 해설을 작성하세요. JSON 형식으로만:\n"
        '{"overall_feedback": "..."}\n\n' + ctx,
        "overall_feedback",
    )
    if not overall:
        return _fallback(card, is_aggregate)

    # (2) 종합 분석을 요약
    summary = await _gen_json(
        _SUMMARY_SYSTEM,
        "다음 종합 해설을 2~3문장으로 요약하세요. JSON 형식으로만:\n"
        '{"summary": "..."}\n\n' + overall,
        "summary",
    )
    logger.success(f"[narrative] ✓ 종합 분석+요약 완료 ({mode})")
    return {"overall_feedback": overall, "summary": summary or overall}
