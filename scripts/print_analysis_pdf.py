"""프론트 /analysis 결과 화면을 그대로 PDF로 인쇄 (백엔드 DOCX 리포트가 아님).

전제: 백엔드(:8000)·프론트(:3000)가 떠 있어야 한다.
흐름: 시스템 Chrome 구동 → 저장된 스코어카드를 sessionStorage 에 주입 →
      /analysis 로 이동(종합 뷰) → 종합 해설 로드 대기 → @media print 적용 → PDF 저장.

Usage:
  python scripts/print_analysis_pdf.py 김멋사
  python scripts/print_analysis_pdf.py 김멋사 --date 2026-02-02   # 단일 일자 뷰
  python scripts/print_analysis_pdf.py 김멋사 --out outputs/김멋사/analysis.pdf
"""
from __future__ import annotations

import argparse
import sys

from playwright.sync_api import sync_playwright

FRONT = "http://localhost:3000"
BACK = "http://localhost:8000"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("instructor")
    p.add_argument("--date", default=None, help="단일 일자(YYYY-MM-DD). 생략 시 종합 뷰")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or f"outputs/{args.instructor}/analysis_{args.date or '종합'}.pdf"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 1400}).new_page()

        # 1) 오리진 확보 후 세션 주입 (백엔드에서 저장 스코어카드 fetch)
        page.goto(FRONT + "/", wait_until="domcontentloaded")
        seeded = page.evaluate(
            """async ({ back, instructor }) => {
                const r = await fetch(back + '/api/v1/analysis/instructor/' + encodeURIComponent(instructor));
                if (!r.ok) return { error: r.status };
                const d = await r.json();
                sessionStorage.setItem('lecture-insight:session', JSON.stringify({
                    instructorId: instructor, scorecards: d.scorecards, source: 'api',
                    createdAt: new Date().toISOString(),
                }));
                return { count: d.scorecards.length };
            }""",
            {"back": BACK, "instructor": args.instructor},
        )
        if seeded.get("error"):
            print(f"[pdf] 스코어카드 조회 실패: {seeded['error']}", file=sys.stderr)
            sys.exit(1)
        print(f"[pdf] 세션 주입 — {seeded['count']}개 강의")

        # 2) 결과 화면으로 (종합이 기본; 단일은 날짜 버튼 클릭)
        page.goto(FRONT + "/analysis", wait_until="networkidle")
        if args.date:
            page.click(f"button:has-text('{args.date[5:].replace('-', '.')}')")
            # 단일 일자 전환 → 해설 재요청. 헤더가 '종합'에서 단일 일자로 바뀔 때까지 대기.
            page.wait_for_timeout(800)
            page.wait_for_function(
                "(d) => { const h = document.querySelector('h1'); "
                "return h && !h.innerText.includes('종합'); }",
                arg=args.date, timeout=20_000,
            )
            page.wait_for_load_state("networkidle")

        # 3) 종합 해설(overall_feedback) 로드 대기 — 긴 단락이 채워질 때까지(최대 80s)
        print("[pdf] 종합 해설 생성 대기 중...")
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll('p')).some(el => (el.innerText||'').length > 150)",
            timeout=80_000,
        )
        page.wait_for_timeout(1500)  # 차트 렌더 안정화

        # 4) 인쇄 스타일 적용 후 PDF
        page.emulate_media(media="print")
        page.pdf(path=out, print_background=True, format="A4",
                 margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"})
        browser.close()
        print(f"[pdf] ✓ 저장 완료 → {out}")


if __name__ == "__main__":
    main()
