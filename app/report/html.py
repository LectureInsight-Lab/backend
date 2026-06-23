"""HTML 리포트 (Jinja2).

레이더/추이 차트를 base64 인라인으로 박아 단일 .html 파일로 출력.
``confidence < threshold`` 항목은 템플릿에서 "⚠ 인간 검토 권장" 으로 표시된다.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.analysis.schemas import InstructorScorecard
from app.report import charts

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def top_items(scorecard: InstructorScorecard, *, best: bool, n: int = 3):
    """final_score 기준 상위(best=True)/하위(best=False) n개 항목."""
    return sorted(scorecard.item_scores, key=lambda it: it.final_score, reverse=best)[:n]


def render(
    scorecard: InstructorScorecard,
    output_path: str,
    radar_b64: str | None = None,
    trend_b64: str | None = None,
) -> str:
    """Scorecard → HTML 파일 경로."""
    radar_b64 = radar_b64 or charts.to_base64(charts.radar_chart(scorecard))
    trend_b64 = trend_b64 or charts.to_base64(charts.trend_chart(scorecard))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")
    rendered = template.render(
        s=scorecard,
        radar_b64=radar_b64,
        trend_b64=trend_b64,
        top_strengths=top_items(scorecard, best=True),
        top_weaknesses=top_items(scorecard, best=False),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    return str(out)
