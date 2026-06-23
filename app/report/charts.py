"""차트 생성 (matplotlib).

레이더 차트: 카테고리별 점수 시각화
추이 차트: 날짜별 종합 점수 시계열

HTML 리포트에는 base64 이미지로 인라인 삽입, DOCX 에는 png bytes 로 삽입.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")        # 헤드리스(서버) 렌더링
import matplotlib.pyplot as plt
import numpy as np

from app.analysis.schemas import InstructorScorecard

# 한국어 라벨 깨짐 방지 — 사용 가능한 한글 폰트로 폴백
for _font in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def radar_chart(scorecard: InstructorScorecard) -> bytes:
    """카테고리별 점수 레이더 차트 → PNG bytes."""
    cats = scorecard.category_scores
    labels = [c.category for c in cats]
    values = [c.score for c in cats]
    if not labels:
        labels, values = ["N/A"], [0.0]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, color="#2563eb", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#2563eb", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("카테고리별 점수", fontsize=13, pad=20)
    return _fig_to_png(fig)


def trend_chart(scorecard: InstructorScorecard) -> bytes:
    """날짜별 종합 점수 추이 차트 → PNG bytes."""
    points = scorecard.trend_points or []
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if points:
        dates = [p.date for p in points]
        scores = [p.score for p in points]
        ax.plot(dates, scores, marker="o", color="#16a34a", linewidth=2)
        for x, y in zip(dates, scores):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "추이 데이터 없음 (단일 강의)", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylim(0, 5)
    ax.set_ylabel("종합 점수")
    ax.set_title("종합 점수 추이", fontsize=13)
    ax.grid(True, alpha=0.3)
    return _fig_to_png(fig)


def to_base64(png_bytes: bytes) -> str:
    """PNG → base64 문자열 (HTML img src 용)."""
    return base64.b64encode(png_bytes).decode("ascii")
