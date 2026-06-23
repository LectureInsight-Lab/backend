"""
notebooks/eda_item12_labels.py — 항목 12: 분모 비교 + 레이블별 SPM 차트

[이수민 - 2026-06-15]
두 가지 추가 탐구를 시각화 → backend/outputs/eda_item12_labels.png:
  (좌) SPM 분모 3종 비교
       - 강의시간(>1800 제외, 현재 spm_kr) ≈ 179
       - 갭캡 30초 (각 갭을 30초로 캡) ≈ 208  ← 제안: spm_kr 개선판
       - 발화시간(≤20초, spm_speaking) ≈ 255  ← 아나운서(355) 비교용
  (우) 개념/예시/실습/기타 레이블별 3분 구간 SPM 분포 (boxplot)

핵심 결론:
  - 갭캡30(208)은 강의시간(179)과 발화(255)의 robust 중간. spm_kr 의 큰-갭 왜곡을 바로잡음.
  - 레이블별 속도 유의미(개념 213 > 예시 206 > 실습 186 > 기타 154, 개념vs실습 p<0.0001).
    → 강사는 '개념'에서 빠르고 '실습'에서 느려짐. 루브릭의 "핵심 개념 감속"과 반대 방향.

실행: python notebooks/eda_item12_labels.py
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from app.core.paths import load_paths, read_stt
from app.preprocessing import pace, preprocessor

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "eda_item12_labels.png")
ANN = pace.ANNOUNCER_SPM
OPTIMAL = (215, 300)

LAB = {
    "개념": ["란 ", "라는", "이라고", "정의", "개념", "의미는", "원리", "이론", "특징", "역할"],
    "예시": ["예를", "예시", "처럼", "가령", "비유", "같은 경우", "예로"],
    "실습": ["코드", "작성", "IDE", "실행", "직접 해", "따라 해", "타이핑", "쳐보", "만들어 보", "켜고", "입력"],
}


def lecture_minutes_capped(utts, cap: int = 30) -> float:
    secs = [u.seconds_from_start for u in utts]
    return sum(min(b - a, cap) for a, b in zip(secs, secs[1:]) if (b - a) >= 0) / 60


def label_window(text: str) -> str:
    score = {k: sum(text.count(w) for w in ws) for k, ws in LAB.items()}
    best = max(score, key=score.get)
    return best if score[best] > 0 else "기타"


def collect():
    files = sorted(glob.glob(os.path.join(str(load_paths().data.stt_dir), "*.txt")))
    kr, cap, spk = [], [], []
    by_label = defaultdict(list)
    for f in files:
        base = os.path.basename(f)
        date, rest = base.split("_", 1)
        utts = preprocessor.parse(read_stt(date, rest.rsplit(".", 1)[0]))
        s = sum(pace.syllable_count(u.text) for u in utts)
        kr.append(pace.spm_kr(utts))
        cap.append(s / lecture_minutes_capped(utts))
        spk.append(pace.spm_speaking(utts))
        # 3분 구간 레이블
        syl, txt = Counter(), defaultdict(list)
        for u in utts:
            b = u.seconds_from_start // 180
            syl[b] += pace.syllable_count(u.text)
            txt[b].append(u.text)
        for b, sc in syl.items():
            if sc >= 90:
                by_label[label_window(" ".join(txt[b]))].append(sc / 3)
    return (kr, cap, spk), by_label


def main() -> None:
    print("집계 ...")
    (kr, cap, spk), by_label = collect()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # (좌) 분모 비교
    names = ["강의시간\n(>1800 제외)", "갭캡 30초\n(제안)", "발화시간\n(≤20초)"]
    data = [kr, cap, spk]
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]
    x = np.arange(3)
    a1.axhspan(*OPTIMAL, color="#5aa469", alpha=0.13)
    a1.axhline(ANN, color="crimson", ls="--", lw=1.5)
    a1.text(2.4, ANN + 5, f"아나운서 {ANN}", color="crimson", fontsize=9, ha="right")
    a1.bar(x, means, yerr=stds, capsize=5, color=["#2f6fb0", "#7a8cc0", "#e0a754"], width=0.6)
    for xi, m in zip(x, means):
        a1.text(xi, m + 7, f"{m:.0f}", ha="center", fontsize=11)
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=9)
    a1.set_ylabel("음절 / 분 (SPM)"); a1.set_ylim(0, 400)
    a1.set_title("① SPM 분모 비교")
    a1.grid(alpha=0.3, axis="y")

    # (우) 레이블별 boxplot
    order = ["개념", "예시", "실습", "기타"]
    box_data = [by_label[l] for l in order]
    colors = ["#2f6fb0", "#5aa469", "#e0a754", "#bdbdbd"]
    bp = a2.boxplot(box_data, labels=[f"{l}\n(n={len(by_label[l])})" for l in order],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for i, l in enumerate(order):
        a2.scatter(i + 1, np.mean(by_label[l]), color="black", zorder=3, s=20)
        a2.text(i + 1, np.mean(by_label[l]) + 12, f"{np.mean(by_label[l]):.0f}", ha="center", fontsize=9)
    # 유의성 (개념 vs 실습)
    t, p = stats.ttest_ind(by_label["개념"], by_label["실습"], equal_var=False)
    y = max(np.percentile(by_label["개념"], 75), np.percentile(by_label["실습"], 75)) + 40
    a2.plot([1, 1, 3, 3], [y, y + 8, y + 8, y], color="black", lw=1)
    a2.text(2, y + 10, f"개념 vs 실습  p={p:.1e} ***", ha="center", fontsize=9)
    a2.set_ylabel("3분 구간 SPM (음절/분)")
    a2.set_title("② 레이블별 발화 속도 — 개념 빠름 / 실습 느림")
    a2.grid(alpha=0.3, axis="y")

    fig.suptitle("항목 12 추가 탐구 — 분모 & 레이블별 속도  (2026-06-15, 이수민)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"저장: {os.path.normpath(OUTPUT_PNG)}")
    print(f"분모: 강의 {means[0]:.0f} / 갭캡30 {means[1]:.0f} / 발화 {means[2]:.0f}")
    print(f"레이블 SPM: " + " / ".join(f"{l} {np.mean(by_label[l]):.0f}" for l in order))
    print(f"개념 vs 실습: t={t:.2f}, p={p:.1e}")


if __name__ == "__main__":
    main()
