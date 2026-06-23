"""
notebooks/eda_item17_engagement.py — 항목 17(참여 유도) EDA + 차트

[이수민 - 2026-06-17]
engagement 를 실제 강의 STT 15개에 적용해 분포 확인:
  (좌) 강의별 engagement_count (유도 발화 수)
  (중) 유도 후 침묵 갭 분류 — '말뿐 vs 진짜 시도 넘김' (<15s/15~60s/1~5분/…)
  (우) 결과 확인 피드백 탐지율 — 진짜(결과확인) vs 습관 tic 포함(되셨어요)
→ backend/outputs/eda_item17_engagement.png 저장.

핵심 관찰(요약은 실행 출력 참조):
  - 유도 평균 9.8회/강의지만 직후 68%가 <15초 → "해보세요" 하고 안 기다림(말뿐).
  - 진짜 시도 넘김(≥30초 침묵)은 ~6%, 진짜 결과 확인 피드백은 ~4%(나머지는 '되셨어요' tic).

실행: python notebooks/eda_item17_engagement.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.paths import load_paths, read_stt
from app.preprocessing import engagement as eng
from app.preprocessing import preprocessor

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "eda_item17_engagement.png")

# 습관 tic 포함 피드백(비교용) — 진짜 결과확인 + '되셨어요/됐어요'
_FB_WITH_TIC = re.compile(r"어떻게\s*됐|결과가|나왔(어요|나요)|완성|에러\s*(없|안)|"
                          r"잘\s*되셨|다\s*하셨|되셨어요|됐어요|되셨나요")
GAP_CLASSES = ["<15s", "15~60s", "1~5분", "5~30분", ">30분"]


def _classify(g: int) -> str:
    if g < 15: return "<15s"
    if g < 60: return "15~60s"
    if g < 300: return "1~5분"
    if g < 1800: return "5~30분"
    return ">30분"


def collect():
    rows, gap_classes = [], {k: 0 for k in GAP_CLASSES}
    for path in sorted(glob.glob(os.path.join(str(load_paths().data.stt_dir), "*.txt"))):
        base = os.path.basename(path)
        d, rest = base.split("_", 1)
        utts = preprocessor.parse(read_stt(d, rest.rsplit(".", 1)[0]))
        prof = eng.engagement_profile(utts)
        prof["date"] = d[5:]
        # 습관 tic 포함 피드백율(비교용)
        hits = prof["hits"]
        tic = sum(1 for h in hits if any(
            _FB_WITH_TIC.search(u.text)
            for u in utts if h.seconds_from_start < u.seconds_from_start <= h.seconds_from_start + 300))
        prof["fb_tic_rate"] = (tic / len(hits) * 100) if hits else 0.0
        for h in hits:
            if h.gap_after is not None:
                gap_classes[_classify(h.gap_after)] += 1
        rows.append(prof)
    return rows, gap_classes


def main() -> None:
    print("15개 파일 항목 17(참여 유도) 집계 ...")
    rows, gap_classes = collect()
    dates = [r["date"] for r in rows]
    counts = [r["engagement_count"] for r in rows]
    genuine = [r["genuine_handoff_count"] for r in rows]
    fb_genuine = [r["feedback_rate"] for r in rows]
    fb_tic = [r["fb_tic_rate"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    # (좌) 강의별 유도 수: 전체 vs 진짜 넘김(≥30초 침묵)
    x = np.arange(len(dates))
    ax1.bar(x - 0.2, counts, width=0.4, color="#bcd0e8", label="유도 발화 수")
    ax1.bar(x + 0.2, genuine, width=0.4, color="#2f6fb0", label="진짜 넘김(≥30초)")
    ax1.set_xticks(x); ax1.set_xticklabels(dates, rotation=90, fontsize=7)
    ax1.set_ylabel("건수")
    ax1.set_title(f"유도 {np.mean(counts):.1f}회 중 진짜 넘김 {np.mean(genuine):.1f}회")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    # (중) 유도 후 갭 분류 (풀링)
    total = sum(gap_classes.values())
    vals = [gap_classes[k] for k in GAP_CLASSES]
    colors = ["#c0504d", "#e0a754", "#5aa469", "#2f6fb0", "#404040"]
    y = np.arange(len(GAP_CLASSES))
    ax2.barh(y, vals, color=colors)
    for yi, v in zip(y, vals):
        ax2.text(v + total * 0.01, yi, f"{v} ({v/total*100:.0f}%)", va="center", fontsize=9)
    ax2.set_yticks(y); ax2.set_yticklabels(GAP_CLASSES, fontsize=9)
    ax2.invert_yaxis(); ax2.set_xlim(0, max(vals) * 1.25)
    ax2.set_xlabel(f"유도 후 침묵 갭 (총 {total}건)")
    ax2.set_title("유도 후 무엇이 일어나나")
    ax2.grid(alpha=0.3, axis="x")

    # (우) 결과 확인 피드백: 진짜 vs 습관 tic 포함
    ax3.bar(x - 0.2, fb_tic, width=0.4, color="#e0a754", label="tic 포함(되셨어요)")
    ax3.bar(x + 0.2, fb_genuine, width=0.4, color="#2f6fb0", label="진짜 결과확인")
    ax3.set_xticks(x); ax3.set_xticklabels(dates, rotation=90, fontsize=7)
    ax3.set_ylabel("피드백 탐지율 (%)")
    ax3.set_title(f"피드백 tic {np.mean(fb_tic):.0f}% → 진짜 {np.mean(fb_genuine):.0f}%")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("항목 17 참여 유도 — EDA  (2026-06-17, 이수민)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"저장: {os.path.normpath(OUTPUT_PNG)}\n")

    # ── 콘솔 요약 ──
    print("── 강의별 ──")
    print(f"{'날짜':>6}{'유도':>4}{'진짜넘김':>6}{'avg_gap':>8}{'피드백%':>7}{'tic%':>6}{'점수':>4}")
    for r in rows:
        print(f"{r['date']:>6}{r['engagement_count']:>4}{r['genuine_handoff_count']:>6}"
              f"{r['avg_gap']:>7.0f}s{r['feedback_rate']:>6.0f}%{r['fb_tic_rate']:>5.0f}%{r['final_score']:>4}")
    base_med = np.mean([r["baseline_gap_median"] for r in rows])
    print("\n── 분포 ──")
    print(f"  engagement_count : 평균 {np.mean(counts):.1f}  범위 {min(counts)}~{max(counts)}  합 {sum(counts)}")
    print(f"  진짜 넘김(≥30초)  : 평균 {np.mean(genuine):.1f}  ({sum(genuine)}/{sum(counts)} = {sum(genuine)/sum(counts)*100:.0f}%)")
    print(f"  avg_gap          : 평균 {np.mean([r['avg_gap'] for r in rows]):.0f}s  vs baseline 라인갭 {base_med:.0f}s")
    print(f"  피드백(진짜)      : 평균 {np.mean(fb_genuine):.0f}%   피드백(tic 포함): 평균 {np.mean(fb_tic):.0f}%")
    print(f"  점수(잠정)        : 평균 {np.mean([r['final_score'] for r in rows]):.1f}  분포 {sorted([r['final_score'] for r in rows])}")
    print("\n── 유도 후 갭 분류(풀링) ──")
    for k in GAP_CLASSES:
        v = gap_classes[k]
        print(f"  {k:<8}: {v:3d} ({v/total*100:4.0f}%)")


if __name__ == "__main__":
    main()
