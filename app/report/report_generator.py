"""리포트 생성 통합 진입점 (6단계).

스코어카드 + 차트 → HTML / DOCX 동시 생성.
차트(레이더/추이)는 한 번만 렌더해 두 포맷이 공유한다.

산출물 위치: ``outputs/{instructor_id}/{lecture_date}.{html,docx}``
"""
from __future__ import annotations

from pathlib import Path

from app.analysis.schemas import InstructorScorecard
from app.report import charts, docx, html

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def generate(
    scorecard: InstructorScorecard,
    formats: list[str] = ("html", "docx"),
    output_root: Path | None = None,
) -> dict[str, str]:
    """선택된 포맷으로 리포트 생성 → {format: file_path}."""
    output_root = output_root or OUTPUT_ROOT
    out_dir = output_root / scorecard.instructor_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scorecard.lecture_date

    # 차트는 1회만 렌더해 공유
    radar_png = charts.radar_chart(scorecard)
    trend_png = charts.trend_chart(scorecard)

    results: dict[str, str] = {}
    if "html" in formats:
        results["html"] = html.render(
            scorecard,
            str(out_dir / f"{stem}.html"),
            radar_b64=charts.to_base64(radar_png),
            trend_b64=charts.to_base64(trend_png),
        )
    if "docx" in formats:
        results["docx"] = docx.render(
            scorecard,
            str(out_dir / f"{stem}.docx"),
            radar_png=radar_png,
            trend_png=trend_png,
        )
    return results
