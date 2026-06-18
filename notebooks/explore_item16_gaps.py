"""
notebooks/explore_item16_gaps.py — 항목 16 탐색: 이해 확인 발화 + 직후 침묵 갭

[이수민 - 2026-06-17] (탐색용 — 정식 모듈 아님)
목적: "되셨어요" 습관 vs 진짜 확인을 가르기 위한 *직후 갭* 신호를 눈으로 확인.
  STT 타임스탬프 = 발화 시작점 → 확인 라인의 gap_after(다음 발화 시작까지)는
  '확인 후 강사가 다음 말을 시작하기까지'의 시간 ≈ 대기/침묵 근사.
  (주의: 학생 마이크 미수록 → 학생 음성 응답은 대개 침묵으로 보임.)

출력:
  1) 샘플 파일의 확인 발화 컨텍스트 (prev → CHECK ⏸gap → next)
  2) 15강의 집계: 확인 직후 갭 분포 vs 전체 라인 갭(baseline) + 갭 임계 통과율

실행: python notebooks/explore_item16_gaps.py [MM-DD]   (기본 02-02)
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.core.paths import load_paths, read_stt
from app.preprocessing import comprehension_check as cc
from app.preprocessing import preprocessor

# 갭 상한(이보다 크면 세션 경계/휴식으로 보고 분석서 제외)
GAP_CAP = 120
# 확인이 라인 '끝'에 있다고 볼 여유 글자수 (이 안에 매칭 끝나면 = 끊고 대기 후보)
TAIL_CHARS = 6


def _hits_with_context(utts):
    """각 확인 발화에 (직후 갭, 라인끝 여부, prev/next 텍스트, 화자변경) 부착."""
    rows = []
    by_idx = {id(u): i for i, u in enumerate(utts)}
    hits = cc.detect_checks(utts)
    # utterance → index 매핑(타임스탬프+텍스트로 다시 찾기보다 순회로)
    idx = 0
    for i, u in enumerate(utts):
        name = cc._first_match(u.text, cc._CHECK_PATTERNS)
        if not name:
            continue
        nxt = utts[i + 1] if i + 1 < len(utts) else None
        prev = utts[i - 1] if i > 0 else None
        gap = (nxt.seconds_from_start - u.seconds_from_start) if nxt else None
        # 매칭이 라인 끝부분인가 (확인하고 끊었나)
        last_pos = max((m.end() for _, rx in cc._CHECK_PATTERNS
                        for m in rx.finditer(u.text)), default=0)
        ends_line = last_pos >= len(u.text.rstrip()) - TAIL_CHARS
        rows.append({
            "ts": u.timestamp, "pattern": name, "text": u.text,
            "gap": gap, "ends_line": ends_line,
            "prev": prev.text if prev else "", "next": nxt.text if nxt else "",
            "spk_change": bool(nxt and nxt.speaker_id != u.speaker_id),
        })
    return rows


def _trunc(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def show_file(date: str, course: str) -> list[dict]:
    utts = preprocessor.parse(read_stt(date, course))
    rows = _hits_with_context(utts)
    valid = [r for r in rows if r["gap"] is not None and r["gap"] <= GAP_CAP]
    print(f"\n{'='*78}\n[{date}] 확인 발화 {len(rows)}건 — 라인끝 확인 직후 갭 큰 순 상위 15\n{'='*78}")
    tail = sorted([r for r in valid if r["ends_line"]], key=lambda r: r["gap"], reverse=True)
    for r in tail[:15]:
        flag = " 👥화자변경" if r["spk_change"] else ""
        print(f"\n {r['ts']}  ⏸{r['gap']:>3}s  [{r['pattern']}]{flag}")
        print(f"    prev: …{_trunc(r['prev'], 48)}")
        print(f"   CHECK: {_trunc(r['text'], 56)}")
        print(f"    next: {_trunc(r['next'], 48)}…")
    return rows


def aggregate():
    print(f"\n{'='*78}\n15강의 집계 — 확인 직후 갭 vs 전체 라인 갭(baseline)\n{'='*78}")
    check_gaps, check_gaps_endline, base_gaps = [], [], []
    n_checks = 0
    for path in sorted(glob.glob(os.path.join(str(load_paths().data.stt_dir), "*.txt"))):
        base = os.path.basename(path)
        d, rest = base.split("_", 1)
        utts = preprocessor.parse(read_stt(d, rest.rsplit(".", 1)[0]))
        secs = [u.seconds_from_start for u in utts]
        base_gaps += [b - a for a, b in zip(secs, secs[1:]) if 0 <= b - a <= GAP_CAP]
        for r in _hits_with_context(utts):
            if r["gap"] is None or r["gap"] > GAP_CAP:
                continue
            n_checks += 1
            check_gaps.append(r["gap"])
            if r["ends_line"]:
                check_gaps_endline.append(r["gap"])

    def pct(a):
        a = np.array(a)
        return f"p50 {np.percentile(a,50):4.1f}s  p75 {np.percentile(a,75):4.1f}s  p90 {np.percentile(a,90):4.1f}s  평균 {a.mean():4.1f}s"

    print(f"\n  전체 라인 갭(baseline)  n={len(base_gaps):5d}  {pct(base_gaps)}")
    print(f"  확인 직후 갭(전체)      n={len(check_gaps):5d}  {pct(check_gaps)}")
    print(f"  확인 직후 갭(라인끝만)  n={len(check_gaps_endline):5d}  {pct(check_gaps_endline)}")

    print("\n  ── 확인 직후 갭 임계 통과율 (라인끝 확인 기준) ──")
    arr = np.array(check_gaps_endline)
    for thr in (3, 5, 8, 12, 20):
        print(f"    gap ≥ {thr:>2}s : {(arr >= thr).mean()*100:5.1f}%  "
              f"({int((arr >= thr).sum())}건)")
    base = np.array(base_gaps)
    print("\n  ── 비교: 전체 라인 갭의 동일 임계 통과율 ──")
    for thr in (3, 5, 8, 12, 20):
        print(f"    gap ≥ {thr:>2}s : {(base >= thr).mean()*100:5.1f}%")
    print(f"\n  → 확인 후 갭이 baseline보다 길면 '대기(진짜 확인)' 신호, 비슷하면 '습관' 신호.")


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "02-02"
    course = "kdt-backendj-21th"
    show_file(f"2026-{date}", course)
    aggregate()


if __name__ == "__main__":
    main()
