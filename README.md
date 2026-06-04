# LectureInsight Backend

강의 스크립트(STT)를 분석해 강사별 강의력 리포트를 생성하는 분석 엔진 API.

## Stack

- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4o / Claude API
- **NLP**: LangChain, KoNLPy, NLTK, pandas
- **Report**: ReportLab (PDF), python-docx (DOCX)

## Project Layout

```
app/
├── api/routes/        # FastAPI 엔드포인트
├── core/              # 설정, 로깅
├── preprocessing/     # 텍스트 정제, 청크 분할, EDA
├── analysis/          # LLM 분석 엔진 (체인, 프롬프트, 스코어링)
│   └── prompts/       # 항목별 프롬프트 (YAML)
├── comparison/        # 강사 비교, 시계열 분석
├── report/            # PDF/DOCX 리포트 생성
├── models/            # Pydantic 데이터 모델
└── utils/
configs/               # 분석 기준, 가중치
data/                  # 원본/처리 데이터 (gitignore)
outputs/               # 생성 리포트 (gitignore)
notebooks/             # EDA, 프롬프트 실험
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # API 키 입력
uvicorn app.main:app --reload
```

## Architecture

### 전체 구조 (Layer View)

```
┌──────────────────────────────────────────────────────────────┐
│                       Client (Frontend)                       │
└──────────────────────────────────────────────────────────────┘
                              │  HTTP / JSON
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  API Layer  (app/api/routes/)                                 │
│  ├─ health.py          GET  /health                           │
│  ├─ analysis.py        POST /api/v1/analysis/lecture          │
│  │                     POST /api/v1/analysis/batch            │
│  │                     GET  /api/v1/analysis/instructor/{id}  │
│  └─ report.py          POST /api/v1/report/generate           │
│                        GET  /api/v1/report/compare            │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Preprocessing  │──▶│   Analysis      │──▶│   Scorer        │
│ (전처리)        │   │  (LLM 체인)      │   │  (점수 집계)     │
│ • cleaner      │   │ • chains.py     │   │ • aggregate()   │
│ • chunker      │   │ • prompts/*.yml │   │   ← weights     │
│ • eda          │   │ • schemas.py    │   │     (yaml)      │
└────────────────┘   └─────────────────┘   └─────────────────┘
                                                    │
                          ┌─────────────────────────┼─────────────────────┐
                          ▼                         ▼                     ▼
                ┌──────────────────┐   ┌─────────────────────┐  ┌──────────────────┐
                │   Comparison     │   │       Report        │  │  JSON Response   │
                │ • instructor.py  │   │ • pdf.py (ReportLab)│  │  (LectureAnalysis│
                │ • timeseries.py  │   │ • docx.py           │  │   Pydantic)      │
                └──────────────────┘   └─────────────────────┘  └──────────────────┘
                                                    │
                                                    ▼
                                            outputs/*.pdf|.docx

┌──────────────────────────────────────────────────────────────┐
│  Core / Config                                                │
│  • core/config.py  (Pydantic Settings, .env 로드)             │
│  • configs/analysis_criteria.yaml (5개 항목 + 가중치)          │
│  • LLM_PROVIDER: openai | anthropic                           │
└──────────────────────────────────────────────────────────────┘
```

### 단일 강의 분석 플로우 (`POST /api/v1/analysis/lecture`)

```mermaid
flowchart TD
    A[Client: STT 스크립트 업로드] --> B[analysis.py: analyze_lecture]
    B --> C[preprocessing/cleaner.clean]
    C -->|공백 정규화, 잡음 제거,<br/>화자/타임스탬프 파싱| D[preprocessing/chunker.split]
    D -->|max_tokens=4000<br/>overlap=200| E[preprocessing/eda.basic_stats]
    E -->|발화량, 어휘 분포| F{analysis_criteria.yaml<br/>5개 항목 로드}

    F --> G1[chains.run_criterion<br/>repetition]
    F --> G2[chains.run_criterion<br/>clarity]
    F --> G3[chains.run_criterion<br/>examples]
    F --> G4[chains.run_criterion<br/>question]
    F --> G5[chains.run_criterion<br/>inappropriate]

    G1 -->|prompts/repetition.yaml| H[LLM<br/>OpenAI/Anthropic]
    G2 -->|prompts/clarity.yaml| H
    G3 -->|prompts/examples.yaml| H
    G4 -->|prompts/question.yaml| H
    G5 -->|prompts/inappropriate.yaml| H

    H -->|Pydantic 구조화 출력| I[CriterionResult x5<br/>score/summary/<br/>evidences/suggestions]

    I --> J[scorer.aggregate]
    J -->|가중치 적용<br/>0.15/0.30/0.20/0.20/0.15| K[LectureAnalysis<br/>overall_score 산출]
    K --> L[JSON 응답]
```

### 배치 / 리포트 플로우

```mermaid
flowchart TD
    subgraph BATCH["POST /api/v1/analysis/batch"]
        B1[다중 스크립트 입력] --> B2[단일 분석 N회 반복]
        B2 --> B3[LectureAnalysis 리스트]
        B3 --> B4[comparison/instructor.compare<br/>강사별 그룹핑]
        B3 --> B5[comparison/timeseries.trend<br/>주차별 변화 추이]
    end

    subgraph REPORT["POST /api/v1/report/generate"]
        R1[LectureAnalysis 입력] --> R2{format?}
        R2 -->|pdf| R3[report/pdf.render<br/>ReportLab]
        R2 -->|docx| R4[report/docx.render<br/>python-docx]
        R3 --> R5[outputs/*.pdf]
        R4 --> R6[outputs/*.docx]
    end
```

### 데이터 흐름 요약 (Pydantic 스키마 기준)

```
raw_text (str)
    ↓ cleaner.clean
cleaned_text (str)
    ↓ chunker.split
chunks (list[str])
    ↓ analysis.chains.run_criterion × 5개 항목
CriterionResult { criterion, score, summary, evidences[], suggestions[] }
    ↓ scorer.aggregate (+ weights)
LectureAnalysis { lecture_id, instructor_id, criteria[], overall_score }
    ↓ ── report.render → PDF/DOCX
    └── comparison.compare/trend → 강사 비교/추이 (배치 시)
```

## Notes

- 실제 강의 데이터는 절대 커밋 금지 (`data/`, `outputs/` 모두 gitignore)
- API 키는 `.env`로만 관리, 절대 코드에 하드코딩 금지
