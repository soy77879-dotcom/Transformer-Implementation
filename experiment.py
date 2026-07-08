import argparse
import csv
import random
import shutil
import warnings
from pathlib import Path

# macOS 기본 Python 환경에서 urllib3/OpenSSL 관련 경고가 출력될 수 있습니다.
# 실험 결과와는 관련이 없으므로 화면을 깔끔하게 하기 위해 숨깁니다.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.util import batch_to_device
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader

from chatbot import (
    FAQ_FILE_PATH,
    MODEL_NAME,
    calculate_text_similarity,
    count_keyword_matches,
    load_faq_data,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = BASE_DIR / "models"
RESULT_FILE_PATH = BASE_DIR / "experiment_results.csv"
DEFAULT_EVALUATION_FILE_PATH = BASE_DIR / "evaluation_questions.csv"


def load_model(model_name):
    """
    SentenceTransformer 모델을 불러옵니다.

    이미 모델이 로컬에 있으면 인터넷 없이 바로 사용하고,
    없으면 Hugging Face에서 다운로드를 시도합니다.
    """
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        return SentenceTransformer(model_name)


def make_question_variants(question):
    """
    FAQ 질문 하나에서 학습/평가에 사용할 간단한 변형 질문을 만듭니다.

    실제 서비스라면 사람이 직접 만든 검증 질문 세트가 가장 좋지만,
    초보 프로젝트에서는 우선 규칙 기반 변형으로 에포크 실험을 시작할 수 있습니다.
    """
    question = str(question).strip()
    variants = set()

    if not question:
        return []

    variants.add(question)

    simple_replacements = [
        ("어떻게 하나요?", "어떻게 해요?"),
        ("어떻게 되나요?", "어떻게 돼요?"),
        ("가능한가요?", "가능해요?"),
        ("하나요?", "하나요?"),
        ("되나요?", "돼요?"),
        ("해야 하나요?", "해야 해요?"),
        ("받을 수 있나요?", "받을 수 있어요?"),
        ("어디서", "어디에서"),
        ("진료 예약", "예약"),
        ("건강검진", "검진"),
    ]

    for old_text, new_text in simple_replacements:
        if old_text in question:
            variants.add(question.replace(old_text, new_text))

    if question.endswith("?"):
        variants.add(question[:-1])

    if "예약" in question:
        variants.add(question.replace("예약", "예약을"))

    if "주차" in question:
        variants.add(question.replace("주차", "주차장"))

    if "입원" in question:
        variants.add(question.replace("입원하려면", "입원 절차는"))

    # 원문과 완전히 같은 문장만 있는 경우도 있으므로 정렬해서 항상 같은 순서로 반환합니다.
    return sorted(variant for variant in variants if variant)


def split_faq_indices(faq_data, train_ratio, random_seed):
    """
    FAQ 행 번호를 학습용과 평가용으로 나눕니다.

    같은 seed를 사용하면 매번 같은 분할이 되어 에포크별 비교가 공정해집니다.
    """
    row_indices = list(faq_data.index)
    random_generator = random.Random(random_seed)
    random_generator.shuffle(row_indices)

    train_size = int(len(row_indices) * train_ratio)
    train_indices = set(row_indices[:train_size])
    eval_indices = set(row_indices[train_size:])

    return train_indices, eval_indices


def build_training_examples(faq_data, train_indices):
    """
    학습용 문장 쌍을 만듭니다.

    모델이 '사용자가 다르게 표현한 질문'과 'FAQ 원문 질문'을 가깝게 보도록
    (변형 질문, 원문 질문) 형태의 positive pair를 만듭니다.
    """
    training_examples = []

    for row_index in train_indices:
        faq_question = str(faq_data.loc[row_index, "question"])
        question_variants = make_question_variants(faq_question)

        for variant_question in question_variants:
            if variant_question != faq_question:
                training_examples.append(InputExample(texts=[variant_question, faq_question]))

    return training_examples


def build_evaluation_queries(faq_data, eval_indices):
    """
    평가용 질문 목록을 만듭니다.

    평가에서는 원문 질문과 너무 똑같은 문장은 제외합니다.
    그래야 모델이 실제 사용자 표현을 얼마나 잘 찾는지 조금 더 현실적으로 볼 수 있습니다.
    """
    evaluation_queries = []

    for row_index in eval_indices:
        faq_question = str(faq_data.loc[row_index, "question"])
        faq_answer = str(faq_data.loc[row_index, "answer"])
        faq_category = str(faq_data.loc[row_index, "category"])
        question_variants = make_question_variants(faq_question)
        acceptable_indices = faq_data[
            (faq_data["answer"].astype(str) == faq_answer)
            & (faq_data["category"].astype(str) == faq_category)
        ].index.tolist()

        for variant_question in question_variants:
            if variant_question != faq_question:
                evaluation_queries.append(
                    {
                        "query": variant_question,
                        "expected_index": row_index,
                        "acceptable_indices": acceptable_indices,
                        "expected_question": faq_question,
                    }
                )

    return evaluation_queries


def build_evaluation_queries_from_file(faq_data, evaluation_file_path):
    """
    사람이 직접 작성한 평가 질문 파일을 읽어 평가용 질문 목록을 만듭니다.

    evaluation_questions.csv는 다음 컬럼을 가져야 합니다.
    - query: 사용자가 입력할 테스트 질문
    - expected_question: 정답으로 기대하는 FAQ 원문 질문
    """
    evaluation_data = pd.read_csv(evaluation_file_path)
    required_columns = {"query", "expected_question"}
    missing_columns = required_columns - set(evaluation_data.columns)

    if missing_columns:
        raise ValueError(f"평가 파일에 필요한 컬럼이 없습니다: {missing_columns}")

    evaluation_queries = []

    for _, evaluation_row in evaluation_data.dropna(subset=["query", "expected_question"]).iterrows():
        query = str(evaluation_row["query"]).strip()
        expected_question = str(evaluation_row["expected_question"]).strip()
        matched_rows = faq_data[faq_data["question"].astype(str) == expected_question]

        if matched_rows.empty:
            raise ValueError(f"FAQ에서 expected_question을 찾지 못했습니다: {expected_question}")

        expected_index = matched_rows.index[0]
        expected_answer = str(faq_data.loc[expected_index, "answer"])
        expected_category = str(faq_data.loc[expected_index, "category"])
        acceptable_indices = faq_data[
            (faq_data["answer"].astype(str) == expected_answer)
            & (faq_data["category"].astype(str) == expected_category)
        ].index.tolist()

        evaluation_queries.append(
            {
                "query": query,
                "expected_index": expected_index,
                "acceptable_indices": acceptable_indices,
                "expected_question": expected_question,
            }
        )

    if not evaluation_queries:
        raise ValueError("평가 파일에서 사용할 수 있는 질문을 찾지 못했습니다.")

    return evaluation_queries


def rank_faq_questions(query, model, faq_data, faq_question_embeddings, use_chatbot_ranking):
    """
    FAQ 후보 순위를 계산합니다.

    에포크 실험에서는 모델 자체의 임베딩 성능을 보는 것이 중요하므로
    기본값은 순수 cosine similarity 순위입니다.

    use_chatbot_ranking=True로 설정하면 챗봇에서 사용하는 글자/키워드 보정까지 포함합니다.
    """
    query_embedding = model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, faq_question_embeddings)[0]

    if not use_chatbot_ranking:
        return semantic_scores.argsort()[::-1]

    ranking_scores = semantic_scores.copy()

    for faq_index, faq_row in faq_data.iterrows():
        text_similarity = calculate_text_similarity(query, str(faq_row["question"]))
        keyword_match_count = count_keyword_matches(
            query,
            str(faq_row["question"]),
            str(faq_row["answer"]),
            str(faq_row["category"]),
        )
        keyword_bonus = min(keyword_match_count * 0.08, 0.24)
        ranking_scores[faq_index] += (text_similarity * 0.35) + keyword_bonus

    ranked_indices = ranking_scores.argsort()[::-1]
    return ranked_indices


def evaluate_model(model, faq_data, evaluation_queries, use_chatbot_ranking):
    """
    모델 성능을 평가합니다.

    - top1_accuracy: 가장 높은 순위의 FAQ가 정답인 비율
    - top3_accuracy: 상위 3개 안에 정답 FAQ가 포함된 비율
    - mean_reciprocal_rank: 정답이 몇 번째에 나왔는지를 반영한 점수
    """
    faq_questions = faq_data["question"].tolist()
    faq_question_embeddings = model.encode(faq_questions)

    if not evaluation_queries:
        raise ValueError("평가용 질문이 없습니다. FAQ 데이터나 변형 규칙을 확인해주세요.")

    top1_correct_count = 0
    top3_correct_count = 0
    reciprocal_rank_total = 0.0

    for evaluation_query in evaluation_queries:
        query = evaluation_query["query"]
        acceptable_indices = set(evaluation_query["acceptable_indices"])
        ranked_indices = rank_faq_questions(
            query,
            model,
            faq_data,
            faq_question_embeddings,
            use_chatbot_ranking,
        )

        if ranked_indices[0] in acceptable_indices:
            top1_correct_count += 1

        if any(ranked_index in acceptable_indices for ranked_index in ranked_indices[:3]):
            top3_correct_count += 1

        rank_position = next(
            position
            for position, ranked_index in enumerate(ranked_indices, start=1)
            if ranked_index in acceptable_indices
        )
        reciprocal_rank_total += 1 / rank_position

    query_count = len(evaluation_queries)

    return {
        "top1_accuracy": top1_correct_count / query_count,
        "top3_accuracy": top3_correct_count / query_count,
        "mean_reciprocal_rank": reciprocal_rank_total / query_count,
        "eval_query_count": query_count,
    }


def train_model_for_epoch_count(model_name, faq_data, train_indices, epoch_count, batch_size, learning_rate):
    """
    지정한 epoch 수만큼 모델을 미세조정합니다.

    epoch_count가 0이면 학습하지 않은 기본 모델을 반환합니다.
    """
    model = load_model(model_name)

    if epoch_count == 0:
        return model

    training_examples = build_training_examples(faq_data, train_indices)

    if not training_examples:
        raise ValueError("학습용 문장 쌍이 없습니다. FAQ 데이터나 변형 규칙을 확인해주세요.")

    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    train_loss.train()

    for epoch_index in range(epoch_count):
        epoch_loss_total = 0.0

        for batch_features, batch_labels in train_dataloader:
            batch_features = [
                batch_to_device(sentence_feature, model.device)
                for sentence_feature in batch_features
            ]
            batch_labels = batch_labels.to(model.device)

            loss_value = train_loss(batch_features, batch_labels)
            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_total += loss_value.item()

        average_epoch_loss = epoch_loss_total / len(train_dataloader)
        print(f"epoch {epoch_index + 1}/{epoch_count} loss={average_epoch_loss:.4f}")

    return model


def save_experiment_results(results):
    """
    하이퍼파라미터 조합별 실험 결과를 CSV 파일로 저장합니다.
    """
    fieldnames = [
        "epoch",
        "batch_size",
        "learning_rate",
        "top1_accuracy",
        "top3_accuracy",
        "mean_reciprocal_rank",
        "eval_query_count",
        "model_path",
    ]

    with RESULT_FILE_PATH.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_epoch_experiment(args):
    """
    여러 하이퍼파라미터 조합을 실험하고 가장 좋은 모델을 저장합니다.
    """
    faq_data = load_faq_data(FAQ_FILE_PATH)
    train_indices, eval_indices = split_faq_indices(faq_data, args.train_ratio, args.seed)
    evaluation_file_path = Path(args.eval_file)

    if not evaluation_file_path.is_absolute():
        evaluation_file_path = BASE_DIR / evaluation_file_path

    if evaluation_file_path.exists():
        evaluation_queries = build_evaluation_queries_from_file(faq_data, evaluation_file_path)
        print(f"평가 파일: {evaluation_file_path}")
    else:
        evaluation_queries = build_evaluation_queries(faq_data, eval_indices)
        print("평가 파일을 찾지 못해 자동 생성 평가 질문을 사용합니다.")

    experiment_results = []
    batch_sizes = args.batch_sizes if args.batch_sizes else [args.batch_size]
    learning_rates = args.learning_rates if args.learning_rates else [args.learning_rate]

    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"전체 FAQ 수: {len(faq_data)}")
    print(f"학습 FAQ 수: {len(train_indices)}")
    print(f"평가 FAQ 수: {len(eval_indices)}")
    print(f"평가 질문 수: {len(evaluation_queries)}")
    print(f"epoch 후보: {args.epochs}")
    print(f"batch size 후보: {batch_sizes}")
    print(f"learning rate 후보: {learning_rates}")

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for epoch_count in args.epochs:
                print(
                    "\n[실험 시작]",
                    f"epoch={epoch_count}",
                    f"batch_size={batch_size}",
                    f"learning_rate={learning_rate}",
                )

                model = train_model_for_epoch_count(
                    MODEL_NAME,
                    faq_data,
                    train_indices,
                    epoch_count,
                    batch_size,
                    learning_rate,
                )

                metrics = evaluate_model(model, faq_data, evaluation_queries, args.use_chatbot_ranking)
                learning_rate_name = str(learning_rate).replace(".", "p").replace("-", "m")
                model_path = MODEL_OUTPUT_DIR / (
                    f"epoch_{epoch_count}_batch_{batch_size}_lr_{learning_rate_name}"
                )
                model.save(str(model_path))

                result_row = {
                    "epoch": epoch_count,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "top1_accuracy": round(metrics["top1_accuracy"], 4),
                    "top3_accuracy": round(metrics["top3_accuracy"], 4),
                    "mean_reciprocal_rank": round(metrics["mean_reciprocal_rank"], 4),
                    "eval_query_count": metrics["eval_query_count"],
                    "model_path": str(model_path),
                }

                experiment_results.append(result_row)

                print(
                    "결과:",
                    f"top1={result_row['top1_accuracy']}",
                    f"top3={result_row['top3_accuracy']}",
                    f"mrr={result_row['mean_reciprocal_rank']}",
                )

    best_result = max(
        experiment_results,
        key=lambda row: (row["top1_accuracy"], row["mean_reciprocal_rank"], row["top3_accuracy"]),
    )
    best_model_path = MODEL_OUTPUT_DIR / "best_model"

    if best_model_path.exists():
        shutil.rmtree(best_model_path)

    shutil.copytree(best_result["model_path"], best_model_path)
    save_experiment_results(experiment_results)

    print("\n[실험 완료]")
    print(f"결과 파일: {RESULT_FILE_PATH}")
    print(
        "가장 좋은 조합:",
        f"epoch={best_result['epoch']}",
        f"batch_size={best_result['batch_size']}",
        f"learning_rate={best_result['learning_rate']}",
    )
    print(f"가장 좋은 모델 경로: {best_model_path}")
    print("챗봇 실행 예시:")
    print(f"python3 chatbot.py --model-path {best_model_path}")


def parse_arguments():
    """
    실험에 사용할 하이퍼파라미터를 터미널 옵션으로 받습니다.
    """
    parser = argparse.ArgumentParser(description="병원 FAQ 챗봇 epoch 실험")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="비교할 epoch 목록입니다. 0은 미세조정 전 기본 모델입니다.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="학습 batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="학습률")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="비교할 batch size 목록입니다. 예: --batch-sizes 8 16",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="비교할 learning rate 목록입니다. 예: --learning-rates 1e-5 2e-5",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="학습 데이터 비율")
    parser.add_argument("--seed", type=int, default=42, help="학습/평가 분할 seed")
    parser.add_argument(
        "--eval-file",
        default=str(DEFAULT_EVALUATION_FILE_PATH),
        help="수동 평가 질문 CSV 파일 경로",
    )
    parser.add_argument(
        "--use-chatbot-ranking",
        action="store_true",
        help="실험 평가에서도 챗봇의 글자/키워드 보정 순위를 사용합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_epoch_experiment(parse_arguments())
