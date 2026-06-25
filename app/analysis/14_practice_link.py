"""
항목 4-2: 실습 연계 (practice_link)

입력: anchored_labeled.json 레코드 (단일 날짜 필터링 후)
  필수 키: anchor_dt, llm_label, text

설계 문서 절차:
  Step 1. llm_label == "실습" 청크 → 실습 이벤트
  Step 2. 실습 직전 마지막 [개념] 청크 (60분 룩백)
  Step 3. similarity = cosine(KR-SBERT(이론), KR-SBERT(실습))
  Step 4. LLM(CoT): 실습 내용이 수강생에게 적절한지 1-5점 평가

채점 (설계 문서 기준):
  sim_score 기준: 5≥0.80 / 4=0.65-0.80 / 2=0.50-0.65 / 1<0.50
  final = round(0.7 × sim_score + 0.3 × llm_score, 0.5단위)

출력:
  {"evidence": list[object] | null, "reason": str, "final_score": int | "N/A"}
  mode=""      → evidence: null
  mode="debug" → evidence: 이론-실습 쌍별 상세 결과
"""
from __future__ import annotations

import json
import os
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

from app.analysis.embedder import cosine_sim, embed

load_dotenv()
_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

_LOOKBACK_MINUTES = 60
_LLM_MODEL        = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

_EVAL_PROMPT = """\
다음은 Java 백엔드 부트캠프 강의의 이론→실습 전환 구간입니다.

[이론 발화]
{theory}

[실습 발화]
{practice}

수강생은 Java 문법은 알지만 고급 프레임워크(Spring, JPA 등)는 초급 수준입니다.

아래 단계로 평가하세요:
1단계: 이론 발화의 핵심 개념 파악
2단계: 실습 발화가 해당 개념을 직접 구현하는지 판단
3단계: 수강생 수준(Java 초급~중급) 적합성 평가
4단계: 1~5점 최종 결정

점수 기준:
5점 — 실습이 이론 개념을 직접 코드로 구현하며 수준에 완전히 적합
4점 — 실습이 이론과 관련되고 대체로 적합
3점 — 관련은 있으나 연결이 명확하지 않거나 수준이 다소 안 맞음
2점 — 실습이 이론과 관계가 약하거나 수준이 크게 안 맞음
1점 — 실습이 이론과 무관하거나 수준이 부적합

JSON으로만 응답하세요:
{{"cot": "1~4단계 추론을 2~3문장으로", "score": 정수, "reason": "최종 판정 한 줄 근거"}}"""


def _llm_score(theory_text: str, practice_text: str) -> tuple[float, str, str]:
    """Gemini CoT 평가. 반환: (score, reason, cot). 실패 시 (3.0, 'LLM 오류', '')."""
    try:
        prompt = _EVAL_PROMPT.format(
            theory=theory_text[:800],
            practice=practice_text[:800],
        )
        resp = _CLIENT.models.generate_content(
            model=_LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        result = json.loads(resp.text)
        score  = max(1.0, min(5.0, float(result["score"])))
        reason = result.get("reason", "")
        cot    = result.get("cot", "")
        return score, reason, cot
    except Exception as e:
        logger.warning(f"[item14] LLM 평가 실패: {e}")
        return 3.0, "LLM 오류", ""


def _sim_to_base_score(sim: float) -> float:
    """설계 문서 기준 similarity → 기본 점수."""
    if sim >= 0.80:
        return 5.0
    if sim >= 0.65:
        return 4.0
    if sim >= 0.50:
        return 2.0
    return 1.0


def score_practice_link(labeled: list[dict] | pd.DataFrame, mode: str = "") -> dict:
    """
    Parameters
    ----------
    labeled : list[dict] or pd.DataFrame
        anchored_labeled.json에서 단일 날짜로 필터링한 레코드.
        필수 컬럼: anchor_dt, llm_label, text
    mode : str
        "" (default) → evidence: null / "debug" → evidence: 쌍별 상세 결과
    """
    df = pd.DataFrame(labeled) if not isinstance(labeled, pd.DataFrame) else labeled.copy()
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
    df = df.sort_values("anchor_dt").reset_index(drop=True)

    practice_df = df[df["llm_label"] == "실습"].reset_index(drop=True)
    concept_df  = df[df["llm_label"] == "개념"].reset_index(drop=True)

    if practice_df.empty:
        logger.info("[item14] 실습 청크 없음 (N/A)")
        return _build(score="N/A", evidence_items=[], reason="실습 레이블 청크 미탐지", mode=mode)

    logger.info(f"[item14] 실습 {len(practice_df)}개, 개념 {len(concept_df)}개 청크 탐지")

    session_scores: list[float] = []
    evidence_items: list[dict]  = []
    cot_parts:      list[str]   = []

    for _, prow in practice_df.iterrows():
        p_dt         = prow["anchor_dt"]
        window_start = p_dt - timedelta(minutes=_LOOKBACK_MINUTES)

        prev_concepts = concept_df[
            (concept_df["anchor_dt"] < p_dt) & (concept_df["anchor_dt"] >= window_start)
        ]
        if prev_concepts.empty:
            logger.debug(f"[item14] {p_dt.strftime('%H:%M')} 실습 — 직전 개념 청크 없음, 건너뜀")
            continue

        theory_row    = prev_concepts.iloc[-1]
        theory_text   = theory_row["text"]
        practice_text = prow["text"]

        if not theory_text.strip() or not practice_text.strip():
            continue

        # Step 3: 코사인 유사도
        embs      = embed([theory_text, practice_text])
        sim       = cosine_sim(embs[0], embs[1])
        sim_score = _sim_to_base_score(sim)

        # Step 4: LLM CoT 평가
        llm_s, llm_reason, llm_cot = _llm_score(theory_text, practice_text)

        # 최종 합산
        raw   = 0.7 * sim_score + 0.3 * llm_s
        final = max(1.0, min(5.0, round(raw * 2) / 2))

        t_label = p_dt.strftime("%H:%M")
        c_label = theory_row["anchor_dt"].strftime("%H:%M")

        session_scores.append(final)
        if llm_cot:
            cot_parts.append(f"[{t_label}] {llm_cot}")

        evidence_items.append({
            "practice_time": t_label,
            "theory_time":   c_label,
            "sim":           round(sim, 3),
            "llm_score":     int(llm_s),
            "llm_reason":    llm_reason,
            "score":         final,
        })

    if not session_scores:
        return _build(
            score="N/A",
            evidence_items=evidence_items,
            reason="유효한 이론-실습 쌍 없음 (직전 개념 청크 부재)",
            mode=mode,
        )

    avg   = sum(session_scores) / len(session_scores)
    score = int(round(max(1.0, min(5.0, round(avg * 2) / 2))))
    reason = "; ".join(cot_parts) if cot_parts else ""

    logger.info(f"[item14] {len(session_scores)}쌍 평균 {avg:.2f} → 최종 {score}")

    return _build(score=score, evidence_items=evidence_items, reason=reason, mode=mode)


def _build(
    score: int | str,
    evidence_items: list[dict],
    reason: str = "",
    mode: str = "",
) -> dict:
    return {
        "evidence":    evidence_items if mode == "debug" else None,
        "reason":      reason,
        "final_score": score,
    }
