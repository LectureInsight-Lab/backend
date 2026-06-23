# 전처리 v2 코드 변경 계획

> 담당: 이수민 (A — 전처리/데이터)
> 브랜치: `preprocessing/v1/sumin`
> 작성일: 2026-06-12
> 근거 문서: 「강의 품질 평가 루브릭 v2.0」(6/9 회의록), 「EDA 항목 정리」(6/11 회의록)
> 상태: **계획 확정 대기** — 코딩 착수 전 팀 공유용

---

## 0. 스코프

6/11 회의록 담당 분배 기준, 본인(이수민)은 **"A — 전처리/데이터"** 담당이며 두 갈래로 일한다.

| 갈래 | 내용 | 다운스트림 |
|---|---|---|
| **(A) 전처리 foundation** | 전원이 쓰는 공통 인프라 — 구간 정의 · 문장 분리 · 형태소 · 시간 윈도우 | 항목 4·5·8·11 등 전원 |
| **(B) 담당 5개 항목** | **2 발화완결성 · 3 언어일관성 · 12 발화속도 · 16 이해확인질문 · 17 참여유도** | 본인 EDA → 임계값 확정 |

### 담당 분배 전문 (6/11)

| 담당자 | 항목 No | 평가 항목 |
|---|---|---|
| **이수민 (A — 전처리/데이터)** | **2, 3, 12, 16, 17** | 발화 완결성, 언어 일관성, 발화 속도, 이해 확인 질문, 참여 유도 |
| 박정현 (D — 리포트/통합) | 1, 6, 7, 11, 14 | 불필요한 반복, 설명 순서, 핵심 내용 강조, 선행 개념 확인, 실습 연계 |

> 🔄 **6/11 갱신 반영**: 항목 3(언어 일관성)이 박정현 → 이수민으로 이관됨. (갱신본 회의록 No 열 기준)
> ⚠️ 단, 갱신본 담당표의 "평가 항목" **이름 열**이 No 열과 불일치(이수민 이름 열에 "언어 일관성" 누락, 박정현 이름 열엔 잔존). **No 열을 정설로 해석**하되 팀에 이름 열 정리 확인 권장.
| 심소민 (C — LLM/루브릭) | 4, 5, 9, 10, 13, 18 | 학습 목표 안내, 전날 복습 연계, 개념 정의, 비유·예시 활용, 예시 적절성, 질문 응답 충분성 |
| 박다현 (B — 분석 엔진) | 8, 15, 18 | 마무리 요약, 오류 대응, 질문 응답 충분성(Q-A 쌍) + 전체 EDA 수합 |

> ⚠️ **소유권 이관**: 현재 `eda.filler_word_ratio`(항목 1)는 박정현 담당으로 이동. `LectureDocument.filler_word_ratio` 필드 소유권은 팀 조율 필요(아래 §6).

---

## 1. Gap 분석 — 현재 코드 ↔ 루브릭 v2.0

| 영역 | 현재 상태 | 루브릭 v2.0 요구 | 변경 |
|---|---|---|---|
| 구간 정의 | `intro=20분 / outro=15분` 고정 (`configs/checklist.yaml`) | `도입부=min(앞 15%, 첫 10분)` · `마무리=min(뒤 10%, 마지막 10분)` | 🔴 로직 교체 |
| 문장 분리 | 없음 | `kss` (항목 2·3 필수) | 🔴 신규 (미설치) |
| 형태소 분석 | `konlpy` 설치만, Mecab 미동작 | 마지막 EF/EC 형태소 판정 (항목 2 완결성 · 3 존댓말/반말) | 🔴 신규 (미설치, 팀 논의) |
| 시간 윈도우 | 없음 | 5분 구간 슬라이싱 (항목 12) | 🔴 신규 |
| 어절/토큰 | `eda._tokenize`(구두점 제거) | 공통정의 `text.split()` | 🟡 정렬 |
| 발화 갭 | `avg_line_gap_seconds` 존재 | 갭 > 10초 = 발화 중단 판정 재사용 (항목 2) | 🟡 확장 |
| 지표 함수 | filler/gap/basic_stats만 | `completeness_rate · consistency_ratio · wpm_kr · check_rate · engagement` | 🔴 5종 신규 |
| EDA 산출물 | **0건** (`notebooks/` = 빈 .gitkeep) | 5개 항목 분포(히스토그램/표준편차/히트맵) | 🔴 신규 |
| 키워드 사전 | `bow_indicators.yaml`에 16/17 일부 | 쉬는시간 · check · engagement · feedback regex | 🟡 확장·이전 |

### 공통 정의 (루브릭 v2.0)

| 용어 | 정의 |
|---|---|
| 어절 | 띄어쓰기 단위 (`text.split()`) |
| 형태소 분석 | KoNLPy 계열 POS 태거 (**팀 논의: Mecab vs Okt vs Komoran** — KcBERT는 형태소 분석기가 아니므로 후보 아님, §5① 참고) |
| 문장 분리 | kss (Korean Sentence Splitter) |
| 임베딩 유사도 | KR-SBERT + cosine (정현·소민 항목용, 본인 직접 사용 X) |
| 도입부 구간 | min(전체 앞 15%, 첫 10분) |
| 마무리 구간 | min(전체 뒤 10%, 마지막 10분) |
| 본문 구간 | 도입부·마무리 제외 나머지 |

---

## 2. 변경 계획 (Phase별 · 파일 단위)

### Phase 0 — 환경/의존성
- [ ] `pyproject.toml`에 **`kss`** 추가.
- [ ] **형태소 분석기 결정** (§5 결정사항 ① — 팀 논의 후). EDA에서 `Mecab vs KcBERT` 비교가 6/11 항목 2에 명시되어 있으므로 최소 하나는 설치 필요.
- [x] `sentence-transformers` 설치 확인됨(KR-SBERT 사용 가능). 본인 항목엔 미사용.
- 환경 확인 결과: `konlpy` ✅ / `sentence_transformers` ✅ / `kss` ❌ / `mecab-ko` ❌

### Phase 1 — 전처리 foundation (갈래 A, **전원 언블로커**)
- [ ] `app/preprocessing/preprocessor.py` — `split_segments` 를 고정분 → **`min(비율, 분캡)`** 로직으로 교체.
- [ ] `configs/checklist.yaml` — `segments` 를 `intro_pct/intro_cap_min`, `outro_pct/outro_cap_min` 형태로 변경. `preprocessor.load_segment_config` 동반 수정.
- [ ] **신규** `app/preprocessing/text_utils.py` — 공통 유틸:
  - `split_sentences(text)` — kss 문장 분리
  - `final_morph(sentence)` — 마지막 종결 형태소(EF/EC) 추출
  - `iter_time_windows(utterances, size_min, overlap)` — N분 슬라이딩 윈도우
- [ ] ⚠️ 구간 정의 변경은 `LectureDocument.intro/outro_lines` 를 바꿔 항목 4·5·8·11 결과에 영향 → **머지 전 팀 공지 필수**.

### Phase 2 — 담당 4개 항목 지표 함수 (갈래 B)
신규 `app/preprocessing/metrics.py` (또는 `eda.py` 확장):

| 항목 | 함수 | 핵심 로직 | 입력 |
|---|---|---|---|
| 2 | `completeness_rate()` | kss 문장분리 → 마지막 형태소 EF=완결/EC·끊김=불완결, 타임스탬프 갭 > 10초 = 발화 중단 | 전사본 + TS |
| 3 | `consistency_ratio()`, `violation_count()` | kss 문장분리 → 마지막 EF 형태소 → 격식존댓말/비격식존댓말/반말/중립 라벨 → 지배 말투 비율 | 전사본 |
| 12 | `wpm_kr()`, `slowdown_ratio()` | 전체 어절수/강의 분, 5분 구간별 WPM·표준편차, 핵심 구간 감속(slowdown < 0.85) | 전사본 + TS(필수) |
| 16 | `check_rate()`, `timing_ratio()` | 이해확인 regex 탐지 → 분당 빈도 + 적절 타이밍(개념/예시/실습 직후 5분 이내) 비율 | 전사본 + TS |
| 17 | `engagement_count()`, `avg_gap()` | 참여유도 regex 탐지 → gap_time(유도→다음 강사 발화) + 확인 피드백 탐지 | 전사본 + TS |

- [ ] 키워드 regex를 **신규 `configs/keywords.yaml`** 로 분리(현 `bow_indicators.yaml` 16/17 항목 이전·확장).

### Phase 3 — EDA 노트북 (현 스프린트 1순위)
신규 `notebooks/eda_preprocessing.ipynb` (또는 항목별 4개) — 15개 강의 파일에 Phase 2 지표 적용:

| 항목 | EDA 할 것 | 기대 산출물 | 임계값 상태 |
|---|---|---|---|
| 2 | 타임스탬프 갭 분포 → 쉬는시간 vs 발화중단 임계값; 쉬는시간 키워드 vs 갭 비교; 형태소 모델 비교 | 갭 분포, completeness_rate 분포 | ⚠️ EDA 후 수정 |
| 3 | 형태소 모델로 존댓말/반말/중립 라벨링 → consistency_ratio 분포 → violation_count 분포 (규칙 vs KcBERT 비교) | consistency_ratio·violation_count 분포 | ⚠️ EDA 후 수정 |
| 12 | 레이블별(개념/비유/실습) WPM_kr 분포·표준편차; slowdown ratio 분포; 150~200 어절/분 적절성 | WPM 분포, 표준편차, slowdown 분포 | ⚠️ EDA 후 수정 |
| 16 | check_count → check_rate → timing_ratio 분포 | check_count/rate, timing_ratio 분포 | ✅ 확정 (0.10회/분 = 5점) |
| 17 | engagement_count → avg_gap → 확인 피드백 탐지율 | gap_time, engagement_count, avg_gap 분포 | ✅ 확정 (2회 = 3점) |

- **출력**: 항목 2·12 임계값 수정안 확정.

### Phase 4 — 스키마·테스트·문서 반영
- [ ] `app/analysis/schemas.py` — 신규 지표 필드 반영(또는 별도 산출물 모델). 문장 단위 정보 추가 여부 결정.
- [ ] `tests/test_preprocessor.py` — 구간 정의 변경 회귀 + 신규 지표 합성 테스트.
- [ ] `docs/preprocessing.md` (공유 repo) — 구간 정의·신규 지표·EDA 결과 갱신.

---

## 3. 담당 4개 항목 상세 (루브릭 v2.0)

### 항목 2 — 발화 완결성 (●중간, LLM ✗)
- **조작적 정의**: 강사 발화 중 종결어미(EF)로 끝나는 완결 문장의 비율. 연결어미(EC)나 호흡 단절로 끊긴 문장은 불완결.
- **측정**: ① 타임스탬프 갭 > **30초**(EDA 확정) = 발화 세그먼트 경계 ② kss 문장 분리 ③ Mecab으로 마지막 형태소 EF/EC 판정 ④ `completeness_rate = 완결 문장 수 / 전체 문장 수 × 100`
- **채점**: ≥95% → 5점 / 90~95% → 4 / 82~90% → 3 / 72~82% → 2 / <72% → 1
- **✅ EDA 결과 (2026-06-15)**: 갭 임계값 **10초→30초 확정**(STT 갭 p50 ~10초 = 발화 길이). 단일 강사 15강의 **completeness_rate 72.2% ± 2.0**. 쉬는시간은 갭·키워드 탐지가 **40%만 일치**(둘 다 단독 불충분) → 완결성은 EF/EC로만 판정, 휴식은 30분 세션분리가 처리. **잠정 밴드: 72%→3점**(팀 협의). 구현: `sentencizer.py`, 차트: `outputs/eda_item02_completeness.png`

### 항목 3 — 언어 일관성 (●중간, LLM ✗ / 보조 KcBERT)
- **조작적 정의**: 강의 전체에서 존댓말 또는 반말 중 지배적 말투의 비율(consistency_ratio). 지배 말투가 아닌 발화 혼용 횟수(violation_count).
- **측정**: ① kss 문장 분리 ② Mecab으로 각 문장 마지막 EF 형태소 추출 ③ 격식존댓말/비격식존댓말/반말/중립 라벨링 ④ `formal_ratio`·`informal_ratio` → `consistency_ratio = max(둘)` ⑤ `violation_count = (전체 - 중립) - 지배 말투 문장 수`
- **보조 모델**: `j5ng/kcbert-formal-classifier` (규칙 기반의 ML 대안 — §5① 결정 B)
- **채점**: ≥97% → 5 / 93~97% → 4 / 85~93% → 3 / 75~85% → 2 / <75% → 1
- **✅ EDA 결과 (2026-06-15)**: 15강의 전부 **존댓말 지배**, **consistency_ratio 72.3% ± 4.6**(64.8~79.3), **violation_count 평균 283**. **한다체(-다)는 중립 처리** — 반말로 세면 일관성 폭락(EF의 ~14%가 -다). 강사가 존댓말 ~72% + 반말(야/어/잖아) ~28% 혼용. **잠정 밴드: 72%→3점**(팀 협의). 구현: `formality.py`, 차트: `outputs/eda_item03_consistency.png`
- **임계값 상태**: ⚠️ → 🟡 EDA 완료, 밴드만 팀 협의 대기 (6/11 항목 3)

### 항목 12 — 발화 속도 적절성 (●중간, LLM ✗)
- **조작적 정의**: 분당 음절 수(SPM)가 적정 범위 내이며, 핵심 개념 설명 구간에서 속도 감소가 관찰되는가. (단위 어절 → **음절(SPM)** 전환, 2026-06-15 팀 안건 통과)
- **측정**: ① `spm_speaking = 전체 음절 수 / 말하는 시간(분)` (헤드라인) ② 5분 구간별 SPM·표준편차 ③ 핵심 구간 `slowdown = 핵심 SPM / 전체 평균 SPM`, < 0.85면 감속. 어절(WPM)은 보조·진단(음절/어절≈2.7).
- **채점** (`score_band`, 아나운서 355 기준): 215~300 + 감속 → 5 / 215~300(균일)·195~215·300~340 → 4 / 160~195·340~390 → 3 / 125~160·390~445 → 2 / <125·>445 → 1
- **✅ EDA 결과 (2026-06-15)**: 실측 발화 SPM **255±6** = **아나운서(355)의 72%** (전 강의 69~74%, 매우 일관), 음절/어절 2.71. 단일 강사 편향은 아나운서 SPM(출판값) 앵커로 제거 — 강사는 천장(355) 아래 최적창(215~300)에 안착 → **score 4점**(slowdown 미연결 → 보수적). 최적창 *위치*는 강의-이해도 연구/전문가 라벨로 확정 예정. 구현 `pace.py`, 차트 `outputs/eda_item12_pace.png`
- **근거**: 아나운서 발화 SPM (jslhd.org, 남 356.9±6.5/여 353.1±4.5) · Kang & Kim (2023, arxiv 2312.00201)

### 항목 16 — 이해 확인 질문 (★높음, LLM ✗)
- **조작적 정의**: 이해 확인 질문의 분당 빈도(check_rate)와 적절 타이밍 비율(timing_ratio).
- **측정**: ① 이해확인 regex 탐지 → check_count ② `check_rate = check_count / 강의 분` ③ 타이밍 OK = 개념정의/예시/실습 종료 이후 5분 이내 → `timing_ratio`
- **채점**: ≥0.10회/분 → 5 / 0.07~0.10 → 4 / 0.05~0.07 → 3 / 0.03~0.05 → 2 / <0.03 → 1. `timing_bonus`: ≥70% +1, <30% -1. `final = min(5, check_rate_score + timing_bonus)`
- **근거**: Rosenshine (2012) — "checked for understanding every 10–15 minutes"

### 항목 17 — 참여 유도 (★높음, LLM ✗)
- **조작적 정의**: 수강생이 직접 시도하도록 유도하는 발화 횟수, 유도 후 실제 시도 시간(침묵 갭), 시도 후 결과 확인 발화 유무.
- **측정**: ① 참여유도 regex → engagement_count ② `gap_time = 다음 강사 발화 - 유도 발화`, avg_gap ③ 확인 피드백 regex(gap 이후 5분 이내)
- **채점**: `engagement_score`(2회↑ →3 / 1회 →2 / 없음 →1) + `gap_bonus`(15초↑ +1) + `feedback_bonus`(있음 +1). `final = min(5, round(합))`
- **근거**: Flanders (1960) — 교사 독점 발화 > 70% = 저품질

---

## 4. 현재 코드 진입점 (참고)

| 파일 | 역할 |
|---|---|
| `app/preprocessing/preprocessor.py` | parse(12h→24h) · split_sessions(30분 갭) · split_segments · build_document |
| `app/preprocessing/eda.py` | filler_word_ratio · avg_line_gap_seconds · basic_stats |
| `app/analysis/schemas.py` | Utterance · Session · LectureDocument 등 전체 파이프라인 스키마 |
| `app/core/paths.py` | load_paths() · read_stt(date, course_id) |
| `configs/checklist.yaml` | 18항목 + `segments` 구간 설정 |
| `configs/bow_indicators.yaml` | BoW 키워드 사전(16/17 일부) |
| `tests/test_preprocessor.py` | 11개 회귀 테스트 |

---

## 5. 결정 필요 사항

### ① 형태소 분석기 (팀 논의 예정)

> ⚠️ **주의 — 결정을 둘로 분리해야 함.** Mecab과 KcBERT는 같은 층위의 도구가 아니다.
> Mecab(Okt/Komoran) = **형태소 분석기**(POS 태거, EF/EC 추출 가능). KcBERT(`j5ng/kcbert-formal-classifier`) = 문장 단위 존댓말/반말 **분류기**(형태소를 보지 않음 → EF/EC 추출 불가).
> 따라서 "Mecab vs KcBERT"는 **항목 2(완결성)** 에서는 성립하지 않는 비교다. 그 비교가 유효한 곳은 항목 3뿐이다.

**결정 A — 형태소 분석기 선택 (foundation, 본인 소유)**
항목 2(완결성, EF/EC) + 공통 정의가 요구. KcBERT는 EF/EC를 못 뽑으므로 후보가 아니다.
- **Okt** (konlpy 내장, 즉시 동작) — 설치 마찰 없이 EDA 곧장 시작, 성능 필요 시 교체
- **Mecab-ko** (brew 설치 필요) — 루브릭 명시 기본값, 빠르고 정확하나 macOS 설치 까다로움
- **Komoran** (konlpy 내장) — 설치 쉬운 또 다른 후보

**결정 B — 항목 3 존댓말/반말 판정 방법 (본인 소유 — 6/11 갱신으로 항목 3 이관됨)**
같은 분류 작업의 두 방법 비교이므로 여기서만 "vs KcBERT"가 타당.
- (a) 규칙 기반: 결정 A의 형태소기로 마지막 EF 추출 → 격식존댓말/비격식존댓말/반말 표 매칭
- (b) ML 기반: KcBERT formal-classifier로 직접 예측

→ **A·B 모두 본인 소유.** A(형태소기)를 먼저 정하면 B의 (a) 규칙 경로가 그 위에 그대로 얹힘. 둘 다 조원들과 추후 논의.

(참고: 형태소 없이 `-습니다/-요` 어미 regex로 항목 2를 처리하는 초경량 fallback도 있으나, `-어`(반말)와 `-어서`(연결어미) 구분이 안 돼 오탐이 많아 루브릭이 형태소 방식을 채택함.)

### ② 🔴 구간 정의 문서 내부 모순 — 반드시 정리
루브릭 v2.0 안에서 도입부 구간 정의가 불일치한다. 코딩 전 canonical 확정 필요:
- **공통 정의**(p.2): 도입부 = `min(앞 15%, 첫 10분)`
- **2-2 전날 복습**(p.7): 도입부 = `min(첫 20분, 앞 25%)`
- **현재 코드**: `첫 20분` 고정

→ 세 값이 전부 다름. **공통 정의를 canonical로 삼을지 결정 필요.**

### ③ `LectureDocument` 스키마 변경 범위
신규 지표(completeness_rate 등)를 `LectureDocument`에 직접 넣을지, 별도 산출물 모델로 분리할지. 다운스트림(정현·소민) 호환 영향.

---

## 6. 진행 순서 (제안)

1. **본 계획 팀 공유** → §5 결정사항 ①②③ 확정
2. Phase 0 (환경) → Phase 1 (foundation, 전원 언블로커)
3. Phase 2 (지표 함수) + Phase 3 (EDA 노트북) — 현 스프린트 1순위
4. Phase 4 (스키마·테스트·문서)
