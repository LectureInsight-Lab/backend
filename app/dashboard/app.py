"""Streamlit 대시보드 (7단계).

실행::

    streamlit run app/dashboard/app.py

기능:
- 강의 분석 실행 트리거 (날짜/강사 선택 또는 STT 직접 입력)
- 결과 조회 (종합/카테고리/항목별 점수)
- 시계열 트렌드 시각화
- 리포트(HTML/DOCX) 생성 및 다운로드 링크
- confidence < 0.5 항목 강조

FastAPI(app.main) 서버가 떠 있어야 한다 (기본 http://localhost:8000).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="LectureInsight 대시보드", layout="wide")
st.title("📊 LectureInsight 강의 분석 대시보드")

with st.sidebar:
    st.header("설정")
    api_base = st.text_input("API Base URL", "http://localhost:8000")
    instructor_id = st.text_input("강사 ID", "instructor_01")
    lecture_date = st.text_input("강의일 (YYYY-MM-DD)", "2026-02-02")
    source = st.radio("STT 입력 방식", ["course_id", "직접 입력"])
    course_id = st.text_input("course_id") if source == "course_id" else None
    raw_text = st.text_area("STT 원문", height=160) if source == "직접 입력" else None


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{api_base}{path}", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"요청 실패: {e}")
        return None


def render_scorecard(card: dict) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("종합 점수", f"{card['overall_score']:.2f} / 5.0")
        if card.get("trend_label"):
            st.caption(f"추이: {card['trend_label']} (slope {card.get('trend_slope', 0):.3f})")
    with c2:
        cat_df = pd.DataFrame(card["category_scores"])
        fig = px.line_polar(cat_df, r="score", theta="category", line_close=True, range_r=[0, 5])
        fig.update_traces(fill="toself")
        st.plotly_chart(fig, use_container_width=True)

    if card.get("trend_points"):
        tdf = pd.DataFrame(card["trend_points"])
        st.plotly_chart(px.line(tdf, x="date", y="score", markers=True, range_y=[0, 5],
                                title="종합 점수 추이"), use_container_width=True)

    st.subheader("18개 항목 상세")
    idf = pd.DataFrame(card["item_scores"])

    def _highlight(row):
        color = "background-color: #fde2e1" if row["needs_human_review"] else ""
        return [color] * len(row)

    cols = ["item_id", "name", "category", "item_type", "final_score", "final_confidence",
            "needs_human_review", "strengths", "improvements"]
    st.dataframe(idf[cols].style.apply(_highlight, axis=1), use_container_width=True, height=560)


tab_run, tab_history = st.tabs(["분석 실행", "강사 이력"])

with tab_run:
    if st.button("▶ 분석 실행", type="primary"):
        payload = {"instructor_id": instructor_id, "lecture_date": lecture_date}
        if course_id:
            payload["course_id"] = course_id
        if raw_text:
            payload["raw_text"] = raw_text
        with st.spinner("분석 중... (Gemini × 18 항목)"):
            card = _post("/api/v1/analysis/lecture", payload)
        if card:
            st.session_state["last_card"] = card
            st.success("분석 완료")

    if "last_card" in st.session_state:
        card = st.session_state["last_card"]
        render_scorecard(card)
        if st.button("📄 리포트 생성 (HTML/DOCX)"):
            res = _post("/api/v1/report/generate",
                        {"scorecard_id": f"{card['instructor_id']}__{card['lecture_date']}"})
            if res:
                for fmt, url in res.get("download", {}).items():
                    st.markdown(f"- [{fmt.upper()} 다운로드]({api_base}{url})")

with tab_history:
    if st.button("🔍 이력 조회"):
        try:
            r = requests.get(f"{api_base}/api/v1/analysis/instructor/{instructor_id}", timeout=60)
            r.raise_for_status()
            hist = r.json()
            st.caption(f"트렌드: {hist['trend_label']} (slope {hist['trend_slope']:.3f})")
            pts = [{"date": c["lecture_date"], "overall": c["overall_score"]} for c in hist["scorecards"]]
            st.plotly_chart(px.line(pd.DataFrame(pts), x="date", y="overall", markers=True,
                                    range_y=[0, 5], title="강사 종합 점수 추이"), use_container_width=True)
            st.json(hist["weekly"])
        except requests.RequestException as e:
            st.error(f"조회 실패: {e}")


def main() -> None:
    """streamlit 엔트리포인트 (호환용 — 스크립트 본문이 UI 를 구성)."""


if __name__ == "__main__":
    main()
