"""DOCX 리포트 (python-docx).

편집 가능한 Word 문서로 출력. 강사/관리자 피드백 협업에 적합.
차트는 PNG bytes 를 임시 스트림으로 삽입한다.
"""
from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

from app.analysis.schemas import InstructorScorecard
from app.report import charts, html


def render(
    scorecard: InstructorScorecard,
    output_path: str,
    radar_png: bytes | None = None,
    trend_png: bytes | None = None,
) -> str:
    """Scorecard → DOCX 파일 경로."""
    radar_png = radar_png or charts.radar_chart(scorecard)
    trend_png = trend_png or charts.trend_chart(scorecard)

    doc = Document()
    doc.add_heading("강의 분석 리포트", level=0)
    p = doc.add_paragraph()
    p.add_run(f"강사 {scorecard.instructor_id} · 강의일 {scorecard.lecture_date}\n").bold = True
    p.add_run(f"생성 {scorecard.analyzed_at.strftime('%Y-%m-%d %H:%M')}")

    # 종합 점수
    doc.add_heading("종합 점수", level=1)
    run = doc.add_paragraph().add_run(f"{scorecard.overall_score:.2f} / 5.0")
    run.font.size = Pt(24)
    run.bold = True
    if scorecard.trend_label:
        doc.add_paragraph(f"추이: {scorecard.trend_label} (slope {scorecard.trend_slope or 0:.3f})")
    doc.add_picture(io.BytesIO(radar_png), width=Inches(4.2))

    # 카테고리
    doc.add_heading("카테고리별 점수", level=1)
    cat_tbl = doc.add_table(rows=1, cols=3)
    cat_tbl.style = "Light Grid Accent 1"
    for h, cell in zip(("카테고리", "점수", "가중치"), cat_tbl.rows[0].cells):
        cell.text = h
    for c in scorecard.category_scores:
        cells = cat_tbl.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = c.category, f"{c.score:.2f}", f"{c.weight*100:.0f}%"

    # 강점 / 개선
    doc.add_heading("강점 Top 3", level=1)
    for it in html.top_items(scorecard, best=True):
        doc.add_paragraph(f"{it.name} — {it.final_score:.2f}점 · {it.strengths}", style="List Bullet")
    doc.add_heading("개선 우선 Top 3", level=1)
    for it in html.top_items(scorecard, best=False):
        doc.add_paragraph(f"{it.name} — {it.final_score:.2f}점 · {it.improvements}", style="List Bullet")

    if scorecard.trend_points:
        doc.add_heading("점수 추이", level=1)
        doc.add_picture(io.BytesIO(trend_png), width=Inches(5.5))

    # 항목 상세
    doc.add_heading("18개 항목 상세", level=1)
    tbl = doc.add_table(rows=1, cols=6)
    tbl.style = "Light Grid Accent 1"
    for h, cell in zip(("ID", "항목", "유형", "점수", "신뢰도", "근거"), tbl.rows[0].cells):
        cell.text = h
    for it in scorecard.item_scores:
        cells = tbl.add_row().cells
        name = it.name + (" ⚠검토" if it.needs_human_review else "")
        cells[0].text = str(it.item_id)
        cells[1].text = name
        cells[2].text = it.item_type
        cells[3].text = f"{it.final_score:.2f}"
        cells[4].text = f"{it.final_confidence:.2f}"
        cells[5].text = it.evidence

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return str(out)
