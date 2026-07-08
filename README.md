# Transformer Implementation Practice

Transformer 이론을 공부한 뒤, 사전학습된 한국어 Transformer 모델을 병원 FAQ 데이터에 적용해본 NLP 실습입니다.

이 저장소는 완성형 챗봇 서비스를 만드는 것보다는, Transformer가 문장을 어떻게 임베딩하고 비슷한 의미의 질문을 어떻게 찾아내는지 직접 확인해보는 데 초점을 두었습니다. 병원 FAQ 데이터를 예제로 사용해 사용자 질문과 FAQ 질문을 비교하고, 가장 가까운 답변을 찾아주는 검색 기반 질의응답 흐름을 구현했습니다.

## What I Practiced

- Transformer 기반 문장 임베딩 사용하기
- 사전학습된 SentenceTransformer 모델을 실제 데이터에 적용하기
- 사용자 질문과 FAQ 질문의 의미 유사도 계산하기
- cosine similarity로 가장 가까운 FAQ 답변 찾기
- Retrieval-based Question Answering 흐름 이해하기
- epoch, batch size, learning rate를 바꿔보며 성능 변화 확인하기
- 실험 결과를 HTML 시각자료로 정리하기

## Tech Stack

| Category | Stack |
| --- | --- |
| Language | Python |
| Data Processing | pandas |
| NLP Model | `jhgan/ko-sroberta-multitask` |
| Library | `sentence-transformers` |
| Similarity | Cosine Similarity |
| Visualization | HTML, SVG |

## Model

| Item | Description |
| --- | --- |
| Model | `jhgan/ko-sroberta-multitask` |
| Architecture | RoBERTa-based Transformer |
| Task Type | Retrieval-based Question Answering |
| Input | 사용자 질문, FAQ 질문 |
| Output | 가장 유사한 FAQ 답변, 카테고리, similarity |

이번 실습에서는 Transformer를 처음부터 학습시키지는 않았습니다. 대신 한국어 문장 의미를 이미 학습한 사전학습 모델을 사용했고, 병원 FAQ 데이터에 맞게 미세조정하면서 성능이 어떻게 달라지는지 확인했습니다.

또한 이 챗봇은 답변을 새로 생성하는 생성형 모델이 아닙니다. FAQ 데이터 안에서 사용자 질문과 가장 의미가 가까운 질문을 찾고, 그에 연결된 답변을 보여주는 검색 기반 구조입니다.

## How It Works

1. `hospital_faq.csv`에서 FAQ 질문, 답변, 카테고리를 불러옵니다.
2. FAQ 질문들을 SentenceTransformer로 문장 임베딩합니다.
3. 사용자가 질문을 입력하면 그 질문도 같은 모델로 임베딩합니다.
4. 사용자 질문과 FAQ 질문 사이의 cosine similarity를 계산합니다.
5. 가장 유사한 FAQ를 선택합니다.
6. similarity가 기준값보다 낮으면 답변을 찾지 못했다는 메시지를 출력합니다.
7. similarity가 기준값 이상이면 답변, 카테고리, similarity를 출력합니다.

## Hyperparameter Experiment

사전학습 모델을 baseline으로 두고, FAQ 데이터에 맞게 미세조정하면서 여러 하이퍼파라미터 조합을 비교했습니다.

| Hyperparameter | Values |
| --- | --- |
| Epoch | 0, 1, 2, 3 |
| Batch Size | 8, 16 |
| Learning Rate | 1e-5, 2e-5 |

평가할 때는 아래 지표를 사용했습니다.

| Metric | Description |
| --- | --- |
| Top-1 Accuracy | 가장 높은 유사도로 선택한 답변이 정답인 비율 |
| Top-3 Accuracy | 상위 3개 후보 안에 정답이 포함된 비율 |
| MRR | 정답이 검색 결과에서 얼마나 앞 순위에 있는지 반영한 지표 |

## Results

실험에서 대표 Best Model로 선택한 조합은 다음과 같습니다.

| Epoch | Batch Size | Learning Rate | Top-1 | Top-3 | MRR |
| --- | --- | --- | --- | --- | --- |
| 2 | 8 | 1e-5 | 1.00 | 1.00 | 1.00 |

이 조합에서 Top-1, Top-3, MRR이 모두 `1.00`으로 가장 안정적인 성능을 보였습니다.

추가로 epoch별 성능을 따로 정리한 실험에서는 `batch_size=16` 조건에서 `epoch=3` 모델이 Top-1 `0.90`, Top-3 `0.95`, MRR `0.9288`로 가장 좋은 결과를 보였습니다.

## Visualization

Transformer 구현 흐름과 하이퍼파라미터 실험 결과는 아래 페이지에서 확인할 수 있습니다.

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

## File Description

| File | Description |
| --- | --- |
| `chatbot.py` | FAQ 챗봇 실행 코드 |
| `experiment.py` | 하이퍼파라미터 실험 코드 |
| `hospital_faq.csv` | FAQ 질문, 답변, 카테고리 데이터 |
| `evaluation_questions.csv` | 평가용 질문 데이터 |
| `experiment_results.csv` | 실험 결과 CSV |
| `hyperparameter_visuals/hyperparameter_experiment_report_standalone.html` | 실험 결과 시각화 HTML |

## Quick Start

챗봇 실행:

```bash
python3 chatbot.py
```

미세조정된 모델로 실행:

```bash
python3 chatbot.py --model-path models/best_model
```

하이퍼파라미터 실험 실행:

```bash
python3 experiment.py
```

## Example Output

```text
질문: 주차장 있나요?
답변: 대부분의 병원은 주차장을 운영하지만, 혼잡할 수 있습니다. 진료 시 주차 할인 여부도 확인해주세요.
카테고리: 주차/교통
유사도: 0.564
```

## Notes

- 이 저장소는 Transformer 이론을 공부한 뒤 실제 코드로 구현해본 실습 결과물입니다.
- 답변을 새로 생성하는 모델이 아니라, FAQ 데이터에서 가장 적절한 답변을 검색하는 구조입니다.
- 모델 파일(`models/`)은 용량이 커서 GitHub 업로드 대상에서 제외했습니다.
- `models/best_model`을 사용하려면 로컬에서 `experiment.py`를 실행해 모델을 생성해야 합니다.
