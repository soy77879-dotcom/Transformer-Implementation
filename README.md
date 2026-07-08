# Hospital FAQ Chatbot

Transformer 기반 문장 임베딩을 활용해 병원 FAQ 질문에 가장 적합한 답변을 찾아주는 챗봇 프로젝트입니다.

이 프로젝트는 단순 키워드 매칭이 아니라, 사용자의 질문과 FAQ 질문을 같은 임베딩 공간에 배치한 뒤 cosine similarity로 의미적으로 가까운 질문을 검색하는 방식으로 구현했습니다.

## 프로젝트 목표

- 병원 FAQ 데이터를 활용한 질의응답형 챗봇 구현
- 사전학습된 한국어 SentenceTransformer 모델 적용
- 사용자 질문과 FAQ 질문 간 의미 유사도 계산
- 유사도가 높은 FAQ 답변, 카테고리, similarity 출력
- epoch, batch size, learning rate를 조정하며 성능 비교
- 하이퍼파라미터 실험 결과를 시각자료로 정리

## 사용한 Transformer 모델

- 모델: `jhgan/ko-sroberta-multitask`
- 라이브러리: `sentence-transformers`
- 기반 구조: RoBERTa 계열 Transformer
- NLP Task: Retrieval-based Question Answering

이 챗봇은 사용자의 질문에 대해 새로운 문장을 생성하는 생성형 QA가 아니라, FAQ 데이터 안에서 가장 적절한 답변을 검색하는 검색 기반 질의응답 구조입니다.

## 구현 방식

1. `hospital_faq.csv` 파일에서 FAQ 질문, 답변, 카테고리를 불러옵니다.
2. FAQ의 모든 질문을 SentenceTransformer로 임베딩합니다.
3. 사용자가 질문을 입력하면 해당 질문도 같은 모델로 임베딩합니다.
4. 사용자 질문 임베딩과 FAQ 질문 임베딩 간 cosine similarity를 계산합니다.
5. 가장 유사한 FAQ를 선택합니다.
6. 유사도가 기준값보다 낮으면 답변을 찾지 못했다는 메시지를 출력합니다.
7. 유사도가 기준값 이상이면 답변, 카테고리, 유사도를 출력합니다.

## 하이퍼파라미터 실험

사전학습 모델을 baseline으로 두고 FAQ 데이터에 맞게 미세조정하면서 다음 하이퍼파라미터를 비교했습니다.

| Hyperparameter | Values |
| --- | --- |
| Epoch | 0, 1, 2, 3 |
| Batch Size | 8, 16 |
| Learning Rate | 1e-5, 2e-5 |

평가 지표는 다음과 같습니다.

| Metric | Meaning |
| --- | --- |
| Top-1 Accuracy | 가장 높은 유사도로 선택한 1개 답변이 정답인 비율 |
| Top-3 Accuracy | 상위 3개 후보 안에 정답이 포함된 비율 |
| MRR | 정답이 검색 결과에서 얼마나 앞 순위에 있는지 반영한 지표 |

## 실험 결과 요약

최고 성능은 여러 조합에서 동률로 나타났고, 대표 Best Model은 다음 조합으로 선택했습니다.

| Epoch | Batch Size | Learning Rate | Top-1 | Top-3 | MRR |
| --- | --- | --- | --- | --- | --- |
| 2 | 8 | 1e-5 | 1.00 | 1.00 | 1.00 |

결과적으로 사전학습 Transformer 모델을 FAQ 데이터에 맞게 미세조정했을 때, `epoch=2`, `batch_size=8`, `learning_rate=1e-5` 조합에서 가장 안정적인 성능을 확인했습니다.

## Hyperparameter Visualization

시각화 자료에서 Transformer 구현 흐름, 하이퍼파라미터 실험 결과, epoch별 성능 변화를 확인할 수 있습니다.

👉 https://soy77879-dotcom.github.io/Transformer-Implementation/

## 주요 파일

| File | Description |
| --- | --- |
| `chatbot.py` | 병원 FAQ 챗봇 실행 코드 |
| `experiment.py` | 하이퍼파라미터 실험 코드 |
| `hospital_faq.csv` | FAQ 질문, 답변, 카테고리 데이터 |
| `evaluation_questions.csv` | 실험 평가용 질문 데이터 |
| `experiment_results.csv` | 하이퍼파라미터 실험 결과 CSV |
| `hyperparameter_visuals/hyperparameter_experiment_report_standalone.html` | 실험 결과 시각화 HTML |
| `hyperparameter_visuals/experiment_results.csv` | 시각자료에 사용한 실험 결과 CSV |

## 실행 방법

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

## 예시 출력

```text
질문: 주차장 있나요?
답변: 대부분의 병원은 주차장을 운영하지만, 혼잡할 수 있습니다. 진료 시 주차 할인 여부도 확인해주세요.
카테고리: 주차/교통
유사도: 0.564
```

## 정리

이 프로젝트는 병원 FAQ 챗봇을 Transformer 기반 Retrieval QA 방식으로 구현하고, 하이퍼파라미터 실험을 통해 어떤 설정에서 FAQ 검색 성능이 가장 좋은지 비교한 프로젝트입니다.
