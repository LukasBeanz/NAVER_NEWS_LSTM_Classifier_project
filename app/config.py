"""네이버 뉴스 LSTM 분류 프로젝트 설정 파일."""

from dataclasses import dataclass

'''
@dataclass 장식자 (decorator : 데코레이터)
데이터를 저장하기 위한 클래스에 주로 사용함

일반적인 class 작성 코드:
class 클래스명:
    def __init__(self) -> None:
        self.필드명 = 초기값
        ......
    def __repr__(self) -> str:
        # 저장된 필드값들을 하나의 문자열(문장)로 만들어서 출력하는 메소드
        .....
    def __eq__(self, other: object) -> bool:
        # 다른 Config 객체 안의 필드값들과 이 객체 안의 필드값들이 모두 일치하는지 확인하는 메소드
        if isinstance(other, Config):
            return self.필드명 == oher.필드명 and .........

=> 클래스 이름위에 @dataclass 표시하면
init(), repr(), eq() 를 자동 생성해 주는 데코레이터임
'''

@dataclass
class Config:
    """학습과 예측에 공통으로 사용하는 하이퍼파라미터를 한 곳에서 관리하는 클래스."""

    # ▼▼▼ [변경] 하이퍼파라미터 전반 상향 조정 (BBC 대비 정확도↑ 손실↓ 목표) ▼▼▼
    # 기존 BBC 설정:
    #   max_vocab=5000, max_len=80, embed_dim=64, hidden_dim=64,
    #   batch_size=8, epochs=8, learning_rate=0.001
    # 변경 이유:
    #   - embed_dim 64→128: 한국어 어절 단위 표현에 더 넓은 벡터 공간 확보
    #   - hidden_dim 64→128: 양방향 LSTM과 조합해 더 풍부한 문맥 표현
    #   - batch_size 8→16: 45건 데이터에서 안정적인 미니배치 학습
    #   - epochs 8→20: 더 충분한 반복 학습으로 수렴 품질 향상
    #   - learning_rate 0.001→0.0005: 과적합 없이 섬세하게 가중치 수렴
    max_vocab: int = 5000           # 토큰화에 사용할 최대 단어 수이다.
    max_len: int = 20               # 한국어 뉴스 제목은 영어 기사보다 짧으므로 80→20으로 줄인다.
    embed_dim: int = 128            # 각 단어를 몇 차원 임베딩 벡터로 바꿀지 정한다. (기존 64 → 128)
    hidden_dim: int = 128           # LSTM 내부 은닉 상태 차원. (기존 64 → 128)
    num_layers: int = 2             # LSTM 층 수. (기존 1층 → 2층, 더 깊은 표현 학습)
    bidirectional: bool = True      # 양방향 LSTM 사용 여부. (기존 단방향 → 양방향)
    batch_size: int = 16            # 한 번의 학습 단계에서 모델에 넣을 샘플 개수. (기존 8 → 16)
    epochs: int = 20                # 전체 데이터를 몇 번 반복 학습할지. (기존 8 → 20)
    learning_rate: float = 0.0005   # Adam 최적화 알고리즘 학습률. (기존 0.001 → 0.0005)
    # ▲▲▲ [변경] 하이퍼파라미터 전반 상향 조정 ▲▲▲

    test_size: float = 0.2          # 평가 데이터 비율. (기존 0.25 → 0.2, 학습 데이터 확보)
    random_state: int = 42          # 실험 결과를 재현하기 위한 난수 고정값이다.

    # ▼▼▼ [변경] 모델 저장 경로 변경 ▼▼▼
    # 기존: "../models/bbc_lstm_model.pt"  (프로젝트 루트 기준으로 상위 폴더를 가리키는 잘못된 경로)
    # 변경: "models/naver_lstm_model.pt"   (프로젝트 루트 기준 올바른 상대 경로, 파일명도 네이버용으로 변경)
    model_path: str = "models/naver_lstm_model.pt"
    # ▲▲▲ [변경] 모델 저장 경로 변경 ▲▲▲

    # ▼▼▼ [추가] 시각화 저장 경로 ▼▼▼
    # 기존: 시각화 없음
    # 변경: 학습 Loss/Accuracy 곡선 그래프를 PNG 파일로 저장하는 경로 추가
    plot_path: str = "models/naver_training_plot.png"
    # ▲▲▲ [추가] 시각화 저장 경로 ▲▲▲
