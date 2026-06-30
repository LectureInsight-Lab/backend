"""scripts/eval_detector.py — 골든셋 대비 detector precision/recall/F1 + 오류 분석.

골든셋(data/gold/item{16,17}_*.json)을 정답으로 보고, production regex detector의
precision·recall·F1을 계산하고 FN(놓침)/FP(헛집계)를 패턴별로 집계한다.

실행: cd backend && python scripts/eval_detector.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.preprocessing import item16_comprehension_check as i16
from app.preprocessing import item17_engagement as i17
from app.preprocessing import sentencizer
from app.preprocessing.preprocessor import parse

DATES = ["02-09", "02-10", "02-11", "02-12", "02-13"]
STT_GLOB = "../data/강의 스크립트/*.txt"
GOLD_DIR = "data/gold"


def _sentences(date):
    f = next(x for x in sorted(glob.glob(STT_GLOB)) if date in x)
    return sentencizer.build_sentences(parse(open(f, encoding="utf-8").read()), use_morph=True)


def _det16(t):  # production 이해확인 detector
    return i16._first_match(t, i16._CHECK_PATTERNS)        # 매칭 패턴명 or None


def _det17(t):  # production 참여유도 detector
    return i17._first_match(t, i17._ENGAGE_PATTERNS)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def eval_item16():
    print("=" * 64)
    print("항목 16 — 이해 확인 질문 (gold: is_check)")
    TP = FP = FN = OUT = 0
    fn_ex, fp_ex = [], []
    for d in DATES:
        sents = _sentences(d)
        gold = {c["idx"]: c["is_check"] for c in json.load(open(f"{GOLD_DIR}/item16_{d}.json", encoding="utf-8"))["candidates"]}
        for i, s in enumerate(sents):
            fired = _det16(s.text) is not None
            if fired and i not in gold:          # net 밖 발화에 firing = 미판정(별도 집계)
                OUT += 1
                continue
            pos = gold.get(i, False)
            if fired and pos:
                TP += 1
            elif fired and not pos:
                FP += 1
                fp_ex.append((d, i, s.text))
            elif (not fired) and pos:
                FN += 1
                fn_ex.append((d, i, s.text))
    p, r, f1 = _prf(TP, FP, FN)
    print(f"  TP {TP} / FP {FP} / FN {FN}  (net 밖 firing {OUT}=미판정)")
    print(f"  precision {p*100:.0f}%  recall {r*100:.0f}%  F1 {f1*100:.0f}%")
    print(f"\n  ▼ FN (놓친 진짜 확인 — recall 개선 대상) {len(fn_ex)}건, 예시:")
    for d, i, t in fn_ex[:10]:
        print(f"    [{d}] {t if len(t)<=54 else '…'+t[-52:]}")
    print(f"\n  ▼ FP (헛집계 — precision 개선 대상) {len(fp_ex)}건, 예시:")
    for d, i, t in fp_ex[:10]:
        print(f"    [{d}] {t if len(t)<=54 else '…'+t[-52:]}")


def eval_item17():
    print("=" * 64)
    print("항목 17 — 참여 유도 (gold: tier strong/weak/no)")
    # 정답 정의 2가지: strict(strong), functional(strong+weak)
    TPs = FPs = FNs = OUT = 0   # strict
    TPf = FNf = 0               # functional recall용
    fn_strong, fp_ex = [], []
    for d in DATES:
        sents = _sentences(d)
        gold = {c["idx"]: c["tier"] for c in json.load(open(f"{GOLD_DIR}/item17_{d}.json", encoding="utf-8"))["candidates"]}
        for i, s in enumerate(sents):
            fired = _det17(s.text) is not None
            if fired and i not in gold:          # net 밖 firing = 미판정
                OUT += 1
                continue
            tier = gold.get(i, "no")
            # strict: 정답 = strong
            if fired and tier == "strong":
                TPs += 1
            elif fired and tier == "no":          # 진짜 헛집계(no에 firing)만 FP
                FPs += 1
                fp_ex.append((d, i, s.text))
            elif (not fired) and tier == "strong":
                FNs += 1
                fn_strong.append((d, i, s.text))
            # functional recall: 정답 = strong∪weak
            if tier in ("strong", "weak"):
                if fired:
                    TPf += 1
                else:
                    FNf += 1
    ps, rs, f1s = _prf(TPs, FPs, FNs)
    rf = TPf / (TPf + FNf) if TPf + FNf else 0.0
    print(f"  [strict: 정답=strong]  TP {TPs} / FP {FPs}(no에 firing) / FN {FNs}  (net 밖 {OUT}=미판정)")
    print(f"    precision {ps*100:.0f}%  recall@strong {rs*100:.0f}%  F1 {f1s*100:.0f}%")
    print(f"  [functional: 정답=strong+weak]  recall {rf*100:.0f}%")
    print(f"\n  ▼ FN strong (놓친 학생 명령형 — recall 개선 핵심) {len(fn_strong)}건, 예시:")
    for d, i, t in fn_strong[:14]:
        print(f"    [{d}] {t if len(t)<=54 else '…'+t[-52:]}")
    print(f"\n  ▼ FP (no에 firing — precision) {len(fp_ex)}건, 예시:")
    for d, i, t in fp_ex[:8]:
        print(f"    [{d}] {t if len(t)<=54 else '…'+t[-52:]}")


if __name__ == "__main__":
    eval_item16()
    print()
    eval_item17()
