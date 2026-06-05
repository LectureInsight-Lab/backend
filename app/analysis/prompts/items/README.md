# 항목별 프롬프트

`configs/checklist.yaml` 의 18개 항목 각각에 대응하는 프롬프트 YAML 을 보관합니다.

## 파일 명명 규칙

`{id:02d}_{snake_name}.yaml` (예: `04_learning_goal.yaml`)

## 작성된 예시

- `01_repetition.yaml` — high_inference, full_sample
- `04_learning_goal.yaml` — discrete, intro

## TODO

나머지 16개 항목 (2, 3, 5~18) 프롬프트 작성.

## 공통 구조

```yaml
id: <int>
name: <str>
category: structure | concept | practice | language | interaction
item_type: discrete | high_inference
context_strategy: intro | middle | outro | full_sample | keyword

description: |
  항목이 무엇을 평가하는지 한 단락 설명.

criteria:
  - 평가 기준 1
  - 평가 기준 2
  - ...

user_template: |
  ... {few_shot_examples} ... {rag_context} ... {positive_count} ... {top_k} ...
```

치환 변수는 `app/analysis/templates.py:build_messages()` 에서 채워집니다.
