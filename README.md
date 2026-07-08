# Transformer Implementation Practice

Transformer 이론을 학습한 뒤, 사전학습된 한국어 Transformer 모델을 병원 FAQ 데이터에 적용해본 NLP 실습 저장소입니다.

이 저장소에서는 `SentenceTransformer`를 활용해 사용자 질문과 FAQ 질문을 문장 임베딩으로 변환하고, cosine similarity 기반으로 가장 유사한 FAQ 답변을 검색합니다. 완성형 서비스보다는 Transformer 기반 문장 임베딩, Retrieval QA, 하이퍼파라미터 실험 과정을 이해하는 데 초점을 두었습니다.

## Overview

- 사전학습 한국어 Transformer 모델 기반 FAQ 검색 구현
- 병원 FAQ 데이터를 활용한 Retrieval-based Question Answering 실습
- 사용자 질문과 FAQ 질문 간 cosine similarity 계산
- 유사도 기준에 따라 답변, 카테고리, similarity 출력
- epoch, batch size, learning rate 조합별 성능 비교
- 실험 결과를 HTML 시각자료로 정리

## Model

| Item | Description |
| --- | --- |
| Model | `jhgan/ko-sroberta-multitask` |
| Library | `sentence-transformers` |
| Architecture | RoBERTa-based Transformer |
| Task | Retrieval-based Question Answering |

이 실습은 Transformer를 처음부터 직접 학습시키는 방식이 아니라, 이미 한국어 문장 의미를 학습한 사전학습 모델을 활용한 뒤 FAQ 데이터에 맞게 미세조정하는 방식으로 진행했습니다.

## How It Works

1. `hospital_faq.csv`에서 FAQ 질문, 답변, 카테고리를 불러옵니다.
2. FAQ 질문을 SentenceTransformer로 문장 임베딩합니다.
3. 사용자가 입력한 질문도 같은 모델로 임베딩합니다.
4. 사용자 질문과 FAQ 질문 간 cosine similarity를 계산합니다.
5. 가장 유사한 FAQ를 선택합니다.
6. similarity가 기준값보다 낮으면 답변을 찾지 못했다는 메시지를 출력합니다.
7. similarity가 기준값 이상이면 답변, 카테고리, similarity를 출력합니다.

## Hyperparameter Experiment

사전학습 모델을 baseline으로 두고, FAQ 데이터에 맞게 미세조정하면서 하이퍼파라미터별 성능 변화를 비교했습니다.

| Hyperparameter | Values |
| --- | --- |
| Epoch | 0, 1, 2, 3 |
| Batch Size | 8, 16 |
| Learning Rate | 1e-5, 2e-5 |

평가에는 다음 지표를 사용했습니다.

| Metric | Description |
| --- | --- |
| Top-1 Accuracy | 가장 높은 유사도로 선택한 답변이 정답인 비율 |
| Top-3 Accuracy | 상위 3개 후보 안에 정답이 포함된 비율 |
| MRR | 정답이 검색 결과에서 얼마나 앞 순위에 있는지 반영한 지표 |

## Results

대표 Best Model은 다음 조합으로 선택했습니다.

| Epoch | Batch Size | Learning Rate | Top-1 | Top-3 | MRR |
| --- | --- | --- | --- | --- | --- |
| 2 | 8 | 1e-5 | 1.00 | 1.00 | 1.00 |

실험 결과, `epoch=2`, `batch_size=8`, `learning_rate=1e-5` 조합에서 Top-1, Top-3, MRR이 모두 `1.00`으로 가장 안정적인 성능을 보였습니다.

## Visualization

하이퍼파라미터 실험 결과와 Transformer 구현 흐름은 아래 페이지에서 확인할 수 있습니다.

👉 https://soy77879-dotcom.github.io/Transformer-Implementation/

## Repository Structure

```text
.
├── chatbot.py
├── experiment.py
├── hospital_faq.csv
├── evaluation_questions.csv
├── experiment_results.csv
└── hyperparameter_visuals/
    ├── experiment_results.csv
    └── hyperparameter_experiment_report_standalone.html
```

## Files

| File | Description |
| --- | --- |
| `chatbot.py` | FAQ 챗봇 실행 코드 |
| `experiment.py` | 하이퍼파라미터 실험 코드 |
| `hospital_faq.csv` | FAQ 질문, 답변, 카테고리 데이터 |
| `evaluation_questions.csv` | 평가용 질문 데이터 |
| `experiment_results.csv` | 실험 결과 CSV |
| `hyperparameter_visuals/hyperparameter_experiment_report_standalone.html` | 실험 결과 시각화 HTML |

## Usage

기본 챗봇 실행:

```bash
python3 chatbot.py
```

미세조정된 모델을 사용할 경우:

```bash
python3 chatbot.py --model-path models/best_model
```

하이퍼파라미터 실험 실행:

```bash
python3 experiment.py
```

## Example

```text
질문: 주차장 있나요?
답변: 대부분의 병원은 주차장을 운영하지만, 혼잡할 수 있습니다. 진료 시 주차 할인 여부도 확인해주세요.
카테고리: 주차/교통
유사도: 0.564
```

## Notes

- 이 저장소는 Transformer 이론 학습 후 구현한 실습 결과물입니다.
- 모델 파일(`models/`)은 용량이 커서 GitHub 업로드 대상에서 제외했습니다.
- `models/best_model`을 사용하려면 로컬에서 `experiment.py`를 실행해 모델을 생성해야 합니다.
