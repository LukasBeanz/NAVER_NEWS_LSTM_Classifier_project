"""PyCharm에서 바로 실행할 수 있는 네이버 뉴스 LSTM 분류 프로젝트 진입점."""

from app.config import Config
from app.predict import load_artifacts, predict_text
from app.train import train_model


if __name__ == "__main__":
    # 프로젝트 전역 설정 객체를 생성한다.
    config = Config()

    # ▼▼▼ [변경] 학습 데이터 및 모델 변경 ▼▼▼
    # 기존: BBC 영어 기사 25건으로 단방향 1층 LSTM 학습
    # 변경: 네이버 뉴스 한국어 기사 45건으로 양방향 2층 LSTM 학습
    #        + 매 epoch 학습/검증 정확도 출력
    #        + 학습 완료 후 Loss/Accuracy 곡선 및 혼동행렬 PNG 저장
    train_model(config)
    # ▲▲▲ [변경] 학습 데이터 및 모델 변경 ▲▲▲

    # 저장된 모델과 단어 사전, 라벨 사전을 다시 불러와 예측 흐름을 확인한다.
    model, metadata = load_artifacts(config)

    # ▼▼▼ [변경] 테스트 기사 문장 변경 ▼▼▼
    # 기존: 영어 BBC 기사 문장
    # 변경: 한국어 네이버 뉴스 스타일 기사 제목으로 교체
    sample_news = "삼성 갤럭시 신형 스마트폰 AI 기능 탑재 출시 발표"
    # ▲▲▲ [변경] 테스트 기사 문장 변경 ▲▲▲

    predicted_label = predict_text(sample_news, model, metadata, config)

    print("\n새 기사:", sample_news)
    print("예측 카테고리:", predicted_label)
