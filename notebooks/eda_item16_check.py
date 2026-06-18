"""
notebooks/eda_item16_check.py — 항목 16(이해 확인 질문) EDA + 차트

[이수민 - 2026-06-17]
comprehension_check 를 실제 강의 STT 15개에 적용해 분포 확인:
  (좌)  강의별 check_rate (분당 확인 빈도) + 채점 밴드(0.03/0.05/0.07/0.10)
  (중)  탐지 패턴 구성(풀링) — 어떤 확인 표현이 실제로 쓰이나
  (우)  timing_ratio 분포(프록시 앵커=예시/실습 cue) + 보너스 경계(30/70%)
→ backend/outputs/eda_item16_check.png 저장.

핵심 관찰(요약은 실행 출력 참조):
  - 이 강사의 확인 마커는 '되셨어요'류 구어 확인이 절대다수. 루브릭 예시 표현은 거의 안 씀.
  - timing_ratio 는 프록시 앵커 기반(정식 앵커=개념정의 4-2/예시·실습 구간 종료는 타 항목 산출물).

실행: python notebooks/eda_item16_check.py
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.paths import load_paths, read_stt
from app.preprocessing import comprehension_check as cc
from app.preprocessing import preprocessor

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "eda_item16_check.png")

# 채점 밴드 경계 (check_rate)
BANDS = [0.03, 0.05, 0.07, 0.10]


def collect():
    rows, pooled_patterns = [], {}
    for path in sorted(glob.glob(os.path.join(str(load_paths().data.stt_dir), "*.txt"))):
        base = os.path.basename(path)
        date, rest = base.split("_", 1)
        utts = preprocessor.parse(read_stt(date, rest.rsplit(".", 1)[0]))
        prof = cc.comprehension_profile(utts, use_proxy_anchors=True)
        prof["date"] = date[5:]          # MM-DD
        rows.append(prof)
        for k, v in prof["pattern_breakdown"].items():
            pooled_patterns[k] = pooled_patterns.get(k, 0) + v
    return rows, pooled_patterns


def main() -> None:
    print("15개 파일 항목 16(이해 확인 질문) 집계 ...")
    rows, patterns = collect()
    dates = [r["date"] for r in rows]
    rates = [r["check_rate"] for r in rows]
    g_rates = [r["genuine_check_rate"] for r in rows]
    counts = [r["check_count"] for r in rows]
    timings = [r["timing_ratio"] for r in rows if r["timing_ratio"] is not None]
    scores = [r["final_score"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    # (좌) 강의별 check_rate: raw(습관 포함) vs genuine(진짜 대기) + 밴드
    x = np.arange(len(dates))
    ax1.bar(x - 0.2, rates, color="#bcd0e8", width=0.4, label="raw (습관 포함)")
    ax1.bar(x + 0.2, g_rates, color="#2f6fb0", width=0.4, label="genuine (진짜 대기)")
    for b, lbl in zip(BANDS, ["2점", "3점", "4점", "5점"]):
        ax1.axhline(b, color="gray", ls="--", lw=0.8)
        ax1.text(len(x) - 0.4, b + 0.001, f"{b}({lbl}↑)", fontsize=7, ha="right", color="gray")
    ax1.set_xticks(x); ax1.set_xticklabels(dates, rotation=90, fontsize=7)
    ax1.set_ylabel("check_rate (회/분)")
    ax1.set_title(f"분당 확인 빈도 raw {np.mean(rates):.3f} → genuine {np.mean(g_rates):.3f}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    # (중) 탐지 패턴 구성 (풀링)
    items = sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    total = sum(vals)
    y = np.arange(len(names))
    ax2.barh(y, vals, color="#2f6fb0")
    for yi, v in zip(y, vals):
        ax2.text(v + total * 0.01, yi, f"{v} ({v/total*100:.0f}%)", va="center", fontsize=8)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(vals) * 1.25)
    ax2.set_xlabel("탐지 수 (15강의 풀링)")
    ax2.set_title(f"확인 표현 구성 (총 {total}회)")
    ax2.grid(alpha=0.3, axis="x")

    # (우) timing_ratio 분포 (프록시 앵커)
    arr = np.array(timings)
    ax3.hist(arr, bins=np.arange(0, 105, 10), color="#e0a754", edgecolor="white")
    ax3.axvspan(70, 100, color="#5aa469", alpha=0.12)
    ax3.axvspan(0, 30, color="#c0504d", alpha=0.10)
    ax3.axvline(float(np.mean(arr)), color="black", ls=":", lw=1.2)
    ax3.text(np.mean(arr) + 1, ax3.get_ylim()[1] * 0.85, f"평균 {np.mean(arr):.0f}%", fontsize=9)
    ax3.text(72, ax3.get_ylim()[1] * 0.5, "+1", color="#2f6f3f", fontsize=9)
    ax3.text(2, ax3.get_ylim()[1] * 0.5, "-1", color="#a03020", fontsize=9)
    ax3.set_xlabel("timing_ratio (%) — 프록시 앵커")
    ax3.set_ylabel("강의 수")
    ax3.set_title("적절 타이밍 비율 분포")
    ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("항목 16 이해 확인 질문 — EDA  (2026-06-17, 이수민)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"저장: {os.path.normpath(OUTPUT_PNG)}\n")

    # ── 콘솔 요약 ──
    print("── 강의별 (raw → genuine) ──")
    print(f"{'날짜':>6} {'분':>6} {'확인':>4} {'진짜':>4} {'raw率':>7} {'진짜率':>7} {'진짜%':>6} {'점수':>4}")
    for r in rows:
        print(f"{r['date']:>6} {r['lecture_minutes']:6.0f} {r['check_count']:4d} "
              f"{r['genuine_count']:4d} {r['check_rate']:7.3f} {r['genuine_check_rate']:7.3f} "
              f"{r['genuine_ratio']:5.0f}% {r['final_score']:4d}")
    g_counts = [r["genuine_count"] for r in rows]
    g_ratios = [r["genuine_ratio"] for r in rows]
    g_scores = [r["genuine_score"] for r in rows]
    print("\n── 분포 ──")
    print(f"  check_count    : 평균 {np.mean(counts):.1f}  범위 {min(counts)}~{max(counts)}")
    print(f"  check_rate(raw): 평균 {np.mean(rates):.3f}  범위 {min(rates):.3f}~{max(rates):.3f}")
    print(f"  진짜 대기 수    : 평균 {np.mean(g_counts):.1f}  범위 {min(g_counts)}~{max(g_counts)}")
    print(f"  genuine_rate   : 평균 {np.mean(g_rates):.3f}  범위 {min(g_rates):.3f}~{max(g_rates):.3f}")
    print(f"  진짜 확인 비율  : 평균 {np.mean(g_ratios):.0f}%  범위 {min(g_ratios):.0f}~{max(g_ratios):.0f}%")
    print(f"  timing_ratio   : 평균 {np.mean(arr):.1f}%  (프록시 앵커)")
    cnt = __import__("collections").Counter
    print(f"  점수 raw       : 평균 {np.mean(scores):.1f}  분포 {dict(sorted(cnt(scores).items()))}")
    print(f"  점수 genuine   : 평균 {np.mean(g_scores):.1f}  분포 {dict(sorted(cnt(g_scores).items()))}")
    print("\n── 확인 표현 구성(풀링) ──")
    for k, v in items:
        print(f"  {k:<12}: {v:4d} ({v/total*100:4.1f}%)")


if __name__ == "__main__":
    main()
