# Hospital FAQ Chatbot

Transformer 기반 병원 FAQ 챗봇 프로젝트입니다.

## 주요 내용

- `jhgan/ko-sroberta-multitask` SentenceTransformer 모델 사용
- FAQ 질문을 문장 임베딩으로 변환
- 사용자 질문과 FAQ 질문 간 cosine similarity 계산
- 가장 유사한 FAQ 답변, 카테고리, 유사도 출력
- epoch, batch size, learning rate를 조정하며 하이퍼파라미터 실험 수행
- 실험 결과를 HTML 시각자료로 정리

## 주요 파일

- `chatbot.py`: 병원 FAQ 챗봇 실행 코드
- `experiment.py`: 하이퍼파라미터 실험 코드
- `hospital_faq.csv`: FAQ 데이터
- `evaluation_questions.csv`: 평가 질문 데이터
- `experiment_results.csv`: 실험 결과 원본 CSV
- `hyperparameter_visuals/hyperparameter_experiment_report_standalone.html`: Transformer 구현 및 실험 결과 시각자료
- `hyperparameter_visuals/experiment_results.csv`: 시각자료에 사용한 실험 결과 CSV

## 실행 방법

```bash
python3 chatbot.py
```

미세조정된 모델을 사용할 경우:

```bash
python3 chatbot.py --model-path models/best_model
```

## 실험 결과 요약

사전학습 Transformer 모델을 FAQ 데이터에 맞게 미세조정한 결과, `epoch=2`, `batch_size=8`, `learning_rate=1e-5` 조합에서 `Top-1`, `Top-3`, `MRR`이 모두 `1.00`으로 가장 안정적인 성능을 보였습니다.
