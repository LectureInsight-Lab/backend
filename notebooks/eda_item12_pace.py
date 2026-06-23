"""
notebooks/eda_item12_pace.py — 항목 12(발화 속도) EDA + 차트 — 음절(SPM) 기준

[이수민 - 2026-06-15 / 팀 안건] 헤드라인 어절 → 음절(SPM) 전환 반영.
pace 를 실제 강의 STT 15개에 적용해:
  (좌) 3가지 SPM 측정(강의/발화/5분 구간) 평균 ± 표준편차 vs 아나운서 355 + 최적창(215~300)
  (우) 5분 구간 SPM 분포(전 파일 풀링)
→ backend/outputs/eda_item12_pace.png 저장.

핵심 결론(2026-06-15):
  - 발화 SPM ≈ 255 = 아나운서(355)의 72% (전 강의 69~74%로 매우 일관). 음절/어절 ≈ 2.7.
  - 아나운서 SPM(출판값)을 외부 앵커로 써 단일 강사 편향을 제거. 강사는 천장(355) 아래
    최적창(215~300)에 안착 → score_band 4점(slowdown 미연결 → 보수적).
  - 최적창의 위치는 강의-이해도 연구/전문가 라벨로 확정 예정.

실행: python notebooks/eda_item12_pace.py
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
from app.preprocessing import pace, preprocessor

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "eda_item12_pace.png")
ANN = pace.ANNOUNCER_SPM           # 355
OPTIMAL = (215, 300)               # score_band 최적창 (아나운서 60~85%)


def collect():
    profs, pooled = [], []
    for path in sorted(glob.glob(os.path.join(str(load_paths().data.stt_dir), "*.txt"))):
        base = os.path.basename(path)
        date, rest = base.split("_", 1)
        utts = preprocessor.parse(read_stt(date, rest.rsplit(".", 1)[0]))
        profs.append(pace.pace_profile(utts))
        pooled += pace.window_spm(utts)
    return profs, pooled


def main() -> None:
    print("15개 파일 SPM 집계 ...")
    profs, pooled = collect()
    measures = {
        "SPM_강의\n(강의시간)": [p["spm_kr"] for p in profs],
        "발화 SPM\n(말하는시간)": [p["spm_speaking"] for p in profs],
        "5분 구간\n중앙값": [p["window_median"] for p in profs],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # (좌) SPM 측정 vs 아나운서 + 최적창
    names = list(measures)
    means = [np.mean(v) for v in measures.values()]
    stds = [np.std(v) for v in measures.values()]
    x = np.arange(len(names))
    ax1.axhspan(*OPTIMAL, color="#5aa469", alpha=0.15)
    ax1.axhline(ANN, color="crimson", ls="--", lw=1.5)
    ax1.text(len(names) - 0.5, ANN + 5, f"아나운서 {ANN}", color="crimson", fontsize=9, ha="right")
    ax1.text(0.0, OPTIMAL[1] - 18, "최적창 215~300\n(score 4~5)", color="#2f6f3f", fontsize=8)
    ax1.bar(x, means, yerr=stds, capsize=5, color=["#2f6fb0", "#e0a754", "#7a8cc0"], width=0.6)
    for xi, m in zip(x, means):
        ax1.text(xi, m + 6, f"{m:.0f}", ha="center", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("음절 / 분 (SPM)")
    ax1.set_ylim(0, 400)
    ax1.set_title("SPM 측정 vs 아나운서 355 + 최적창")
    ax1.grid(alpha=0.3, axis="y")

    # (우) 5분 구간 SPM 분포 (풀링)
    arr = np.array(pooled)
    ax2.hist(arr, bins=np.arange(80, 400, 20), color="#e0a754", edgecolor="white")
    ax2.axvspan(*OPTIMAL, color="#5aa469", alpha=0.15)
    ax2.axvline(ANN, color="crimson", ls="--", lw=1.5)
    ax2.axvline(np.median(arr), color="black", ls=":", lw=1.2)
    ax2.text(np.median(arr) + 4, ax2.get_ylim()[1] * 0.9, f"중앙값 {np.median(arr):.0f}", fontsize=9)
    ax2.set_title("5분 구간 SPM 분포 (15강의 풀링)")
    ax2.set_xlabel("음절 / 분 (SPM)")
    ax2.set_ylabel("구간 수")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("항목 12 발화 속도 — EDA (음절 SPM)  (2026-06-15, 이수민)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")

    print(f"저장: {os.path.normpath(OUTPUT_PNG)}")
    for n, v in measures.items():
        print(f"  {n.replace(chr(10), ' '):<22}: 평균 {np.mean(v):.1f}  범위 {min(v):.1f}~{max(v):.1f}")
    spk = [p["spm_speaking"] for p in profs]
    print(f"  발화 SPM = 아나운서의 {np.mean(spk)/ANN*100:.0f}%  / score 전부 {profs[0]['score']}점(보수적)")


if __name__ == "__main__":
    main()
