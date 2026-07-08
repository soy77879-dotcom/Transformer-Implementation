import argparse
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
import warnings

# macOS 기본 Python 환경에서 urllib3/OpenSSL 관련 경고가 출력될 수 있습니다.
# 챗봇 동작에는 직접적인 문제가 없으므로 사용자 화면을 깔끔하게 하기 위해 숨깁니다.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 유사도 기준값입니다.
# 가장 비슷한 FAQ 질문과의 유사도가 이 값보다 낮으면 답변을 찾지 못한 것으로 처리합니다.
SIMILARITY_THRESHOLD = 0.45


# chatbot.py 파일이 있는 폴더 경로입니다.
# 이 값을 사용하면 터미널의 현재 위치가 달라도 hospital_faq.csv를 안정적으로 찾을 수 있습니다.
BASE_DIR = Path(__file__).resolve().parent


# FAQ 데이터가 저장된 CSV 파일 경로입니다.
FAQ_FILE_PATH = BASE_DIR / "hospital_faq.csv"


# 한국어 문장 임베딩에 사용할 SentenceTransformer 모델 이름입니다.
MODEL_NAME = "jhgan/ko-sroberta-multitask"


def normalize_text(text):
    """
    비교하기 쉬운 형태로 문장을 정리합니다.

    예를 들어 "토요일에도 진료 하나요?"와 "토요일에도 진료하나요?"는
    띄어쓰기만 다를 뿐 거의 같은 질문입니다.
    이런 경우를 잘 잡기 위해 공백과 문장부호를 제거합니다.

    또한 "주차장"과 "주차", "에약"과 "예약"처럼 실제 사용자가 자주 쓰는
    표현 차이와 오타를 간단히 정규화합니다.
    """
    normalized_text = str(text).lower()

    synonym_replacements = {
        "에약": "예약",
        "주차장": "주차",
        "문여나요": "진료하나요",
        "문 여나요": "진료하나요",
        "돼요": "되나요",
        "되요": "되나요",
        "해요": "하나요",
    }

    for original_text, replacement_text in synonym_replacements.items():
        normalized_text = normalized_text.replace(original_text, replacement_text)

    return "".join(character for character in normalized_text if character.isalnum())


def calculate_text_similarity(user_question, faq_question):
    """
    사용자 질문과 FAQ 질문의 글자 기반 유사도를 계산합니다.

    SentenceTransformer의 의미 기반 유사도는 문맥을 잘 보지만,
    짧은 문장에서 띄어쓰기나 어미가 조금 달라지면 점수가 낮게 나올 수 있습니다.
    이 함수는 그런 짧은 FAQ 질문을 보완하기 위한 보조 점수입니다.
    """
    normalized_user_question = normalize_text(user_question)
    normalized_faq_question = normalize_text(faq_question)

    if not normalized_user_question or not normalized_faq_question:
        return 0.0

    return SequenceMatcher(None, normalized_user_question, normalized_faq_question).ratio()


def calculate_confidence_score(semantic_similarity, text_similarity, keyword_match_count):
    """
    사용자에게 보여줄 최종 유사도를 계산합니다.

    semantic_similarity는 SentenceTransformer가 계산한 의미 기반 점수이고,
    text_similarity는 띄어쓰기와 문장부호를 제거한 글자 기반 점수입니다.
    keyword_match_count는 핵심 단어가 얼마나 겹치는지 나타냅니다.

    답변 후보를 고를 때는 여러 보정 점수를 사용하지만,
    사용자에게 보여주는 유사도는 너무 과하게 높아 보이지 않도록 더 보수적으로 계산합니다.

    같은 의미의 질문이어도 표현이 다르면 0.7~0.8대가 나오게 하고,
    거의 같은 문장일 때만 0.9대가 나오도록 조정합니다.
    """
    keyword_score = min(keyword_match_count / 3, 1.0)

    if text_similarity >= 0.98:
        return 0.90

    calibrated_similarity = (
        (semantic_similarity * 0.65)
        + (text_similarity * 0.15)
        + (keyword_score * 0.05)
    )

    # 핵심 키워드가 하나라도 정확히 겹치면 답변 기준값 아래로 떨어지지 않도록 합니다.
    # 예: "주차장 있나요?"는 FAQ의 "주차는 가능한가요?"와 표현은 다르지만 핵심은 같습니다.
    if keyword_match_count > 0:
        calibrated_similarity = max(calibrated_similarity, 0.55)

    return min(0.90, calibrated_similarity)


def count_keyword_matches(user_question, faq_question, faq_answer, faq_category):
    """
    사용자 질문과 FAQ가 같은 핵심 단어를 얼마나 공유하는지 계산합니다.

    예를 들어 사용자가 "전화로 예약 가능한가요?"라고 질문하면
    cosine similarity만으로는 "전화로 처방받을 수 있나요?"가 더 비슷하게 나올 수 있습니다.
    이 함수는 "예약"처럼 의미를 결정하는 중요한 단어가 같은 FAQ를 조금 더 우선하도록 도와줍니다.
    """
    important_keywords = [
        "예약",
        "취소",
        "변경",
        "접수",
        "진료",
        "시간",
        "토요일",
        "일요일",
        "공휴일",
        "응급",
        "진료비",
        "결제",
        "보험",
        "서류",
        "진단서",
        "소견서",
        "처방",
        "검진",
        "내시경",
        "검사",
        "MRI",
        "CT",
        "X-ray",
        "초음파",
        "예방접종",
        "코로나",
        "소아",
        "산부인과",
        "피부과",
        "정형외과",
        "입원",
        "절차",
        "면회",
        "수술",
        "주차",
        "셔틀",
        "위치",
        "휠체어",
        "외국인",
        "통역",
        "신분증",
        "기록",
        "의뢰서",
        "비급여",
        "대기",
        "약국",
        "마스크",
    ]

    faq_text = f"{faq_question} {faq_answer} {faq_category}"

    normalized_user_question = normalize_text(user_question)
    normalized_faq_text = normalize_text(faq_text)

    match_count = 0
    for keyword in important_keywords:
        normalized_keyword = normalize_text(keyword)
        if normalized_keyword in normalized_user_question and normalized_keyword in normalized_faq_text:
            match_count += 1

    return match_count


def load_faq_data(file_path):
    """
    hospital_faq.csv 파일을 읽어서 FAQ 데이터를 불러옵니다.

    CSV 파일에는 반드시 다음 컬럼이 있어야 합니다.
    - question: FAQ 질문
    - answer: FAQ 답변
    - category: FAQ 분류
    """
    if not file_path.exists():
        raise FileNotFoundError(f"FAQ CSV 파일을 찾을 수 없습니다: {file_path}")

    faq_data = pd.read_csv(file_path)

    # 프로그램이 올바르게 동작하려면 필요한 컬럼이 모두 있어야 하므로 미리 확인합니다.
    required_columns = {"question", "answer", "category"}
    missing_columns = required_columns - set(faq_data.columns)

    if missing_columns:
        raise ValueError(f"CSV 파일에 필요한 컬럼이 없습니다: {missing_columns}")

    # question, answer, category 컬럼에 빈 값이 있으면 오류나 이상한 답변이 생길 수 있으므로 제거합니다.
    faq_data = faq_data.dropna(subset=["question", "answer", "category"]).reset_index(drop=True)

    if faq_data.empty:
        raise ValueError("사용할 수 있는 FAQ 데이터가 없습니다.")

    return faq_data


def load_embedding_model(model_name):
    """
    문장을 숫자 벡터로 바꿔 주는 SentenceTransformer 모델을 불러옵니다.

    이미 모델이 다운로드되어 있으면 로컬 캐시에서 바로 불러옵니다.
    로컬 캐시에 모델이 없으면 인터넷을 통해 Hugging Face에서 다운로드를 시도합니다.
    """
    try:
        # 모델이 이미 캐시에 있으면 네트워크 확인 없이 바로 불러옵니다.
        # 이렇게 하면 인터넷 연결이 불안정하거나 차단된 환경에서도 실행이 빨라집니다.
        embedding_model = SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        print("로컬에 모델이 없어 다운로드를 시도합니다. 처음 실행 시 시간이 걸릴 수 있습니다.")
        embedding_model = SentenceTransformer(model_name)

    return embedding_model


def create_question_embeddings(embedding_model, faq_questions):
    """
    FAQ의 모든 question을 미리 임베딩합니다.

    사용자가 질문할 때마다 FAQ 전체를 다시 임베딩하면 느리기 때문에,
    프로그램 시작 시 FAQ 질문 임베딩을 한 번만 만들어 둡니다.
    """
    question_embeddings = embedding_model.encode(faq_questions)
    return question_embeddings


def find_most_similar_faq(user_question, embedding_model, faq_data, faq_question_embeddings):
    """
    사용자의 질문과 가장 유사한 FAQ를 찾습니다.

    처리 과정:
    1. 사용자 질문을 임베딩합니다.
    2. 사용자 질문 임베딩과 FAQ 질문 임베딩들의 cosine similarity를 계산합니다.
    3. 가장 유사도가 높은 FAQ의 위치와 유사도 값을 구합니다.
    4. 해당 FAQ 행과 유사도 값을 반환합니다.
    """
    user_question_embedding = embedding_model.encode([user_question])

    similarity_scores = cosine_similarity(user_question_embedding, faq_question_embeddings)

    # similarity_scores는 [[0.1, 0.7, 0.3]]처럼 2차원 배열 형태입니다.
    # 첫 번째 행만 꺼내면 FAQ 질문별 유사도 목록을 얻을 수 있습니다.
    similarity_scores = similarity_scores[0]

    ranking_scores = similarity_scores.copy()
    confidence_scores = similarity_scores.copy()

    # cosine similarity만 사용하면 "전화로 예약 가능한가요?"처럼 여러 의미가 섞인 질문에서
    # "전화" 단어 때문에 처방 FAQ가 선택될 수 있습니다.
    # 핵심 키워드가 함께 등장하는 FAQ에 작은 보너스를 더해 더 자연스러운 답변을 고릅니다.
    for faq_index, faq_row in faq_data.iterrows():
        text_similarity = calculate_text_similarity(user_question, str(faq_row["question"]))
        keyword_match_count = count_keyword_matches(
            user_question,
            str(faq_row["question"]),
            str(faq_row["answer"]),
            str(faq_row["category"]),
        )

        keyword_bonus = min(keyword_match_count * 0.08, 0.24)

        # 답변 후보를 고를 때는 의미 기반 점수, 글자 기반 점수, 핵심 키워드 점수를 함께 봅니다.
        ranking_scores[faq_index] += (text_similarity * 0.35) + keyword_bonus

        # 사용자에게 보여줄 유사도는 0~1 사이의 confidence 점수로 계산합니다.
        # 의미 기반 점수가 낮아도 질문 문장이 거의 같으면 높은 점수가 나오도록 보정합니다.
        confidence_scores[faq_index] = calculate_confidence_score(
            similarity_scores[faq_index],
            text_similarity,
            keyword_match_count,
        )

    most_similar_index = ranking_scores.argmax()
    highest_similarity = confidence_scores[most_similar_index]

    most_similar_faq = faq_data.iloc[most_similar_index]

    return most_similar_faq, highest_similarity


def print_chatbot_answer(most_similar_faq, similarity):
    """
    유사도 기준에 따라 답변을 출력합니다.

    유사도가 기준값보다 낮으면 답변을 찾지 못했다는 메시지를 출력하고,
    기준값 이상이면 FAQ의 answer, category, similarity를 출력합니다.
    """
    if similarity < SIMILARITY_THRESHOLD:
        print("죄송합니다. 해당 질문에 대한 답변을 찾지 못했습니다.")
        return

    print(f"답변: {most_similar_faq['answer']}")
    print(f"카테고리: {most_similar_faq['category']}")
    print(f"유사도: {similarity:.3f}")


def run_chatbot(model_name=MODEL_NAME):
    """
    병원 FAQ 챗봇을 실행하는 메인 함수입니다.

    사용자가 질문을 입력하면 가장 비슷한 FAQ를 찾아 답변하고,
    사용자가 '종료'를 입력하면 프로그램을 끝냅니다.
    """
    faq_data = load_faq_data(FAQ_FILE_PATH)
    embedding_model = load_embedding_model(model_name)

    faq_questions = faq_data["question"].tolist()
    faq_question_embeddings = create_question_embeddings(embedding_model, faq_questions)

    print("병원 FAQ 챗봇입니다.")
    print("질문을 입력해주세요. 프로그램을 끝내려면 '종료'를 입력하세요.")

    while True:
        user_question = input("\n질문: ").strip()

        if user_question == "종료":
            print("챗봇을 종료합니다.")
            break

        if not user_question:
            print("질문을 입력해주세요.")
            continue

        most_similar_faq, similarity = find_most_similar_faq(
            user_question,
            embedding_model,
            faq_data,
            faq_question_embeddings,
        )

        print_chatbot_answer(most_similar_faq, similarity)


def parse_arguments():
    """
    터미널에서 챗봇 실행 옵션을 받습니다.

    기본값은 사전학습 모델이지만, 실험으로 저장한 모델 경로를 넘기면
    해당 모델을 사용해서 챗봇을 실행할 수 있습니다.
    """
    parser = argparse.ArgumentParser(description="병원 FAQ 챗봇 실행")
    parser.add_argument(
        "--model-path",
        default=MODEL_NAME,
        help="사용할 SentenceTransformer 모델 이름 또는 로컬 모델 폴더 경로",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_chatbot(args.model_path)
