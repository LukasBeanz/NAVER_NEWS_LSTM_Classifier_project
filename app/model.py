"""PyTorch 기반 LSTM 기사 분류 모델 정의 모듈."""

from __future__ import annotations

import torch
from torch import nn


class TextLSTMClassifier(nn.Module):
    """Embedding, LSTM, Dropout, Linear 계층으로 구성된 텍스트 분류 모델.

    # ▼▼▼ [변경] 양방향·다층 LSTM 지원 추가 ▼▼▼
    # 기존: 단방향 1층 LSTM 고정
    # 변경: num_layers, bidirectional 파라미터 추가
    #        - bidirectional=True → 문장 앞→뒤, 뒤→앞 두 방향 동시 학습
    #        - num_layers=2       → LSTM을 2층으로 쌓아 더 깊은 패턴 학습
    #        → BBC 모델 대비 정확도↑ 손실↓ 기대
    # ▲▲▲ [변경] 양방향·다층 LSTM 지원 추가 ▲▲▲
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,          # ← [추가] 기존: 하드코딩 1층. 변경: 파라미터로 외부에서 지정 가능
        bidirectional: bool = False,  # ← [추가] 기존: 단방향 고정.  변경: True 시 양방향 LSTM 사용
    ) -> None:
        """모델에 필요한 계층을 생성한다."""

        super().__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ▼▼▼ [변경] LSTM 생성자에 num_layers, bidirectional 파라미터 추가 ▼▼▼
        # 기존: nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 변경: num_layers, bidirectional 을 외부 파라미터로 받아 유연하게 구성
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0.0,  # 다층 LSTM 층 사이 dropout으로 과적합 방지
        )
        # ▲▲▲ [변경] LSTM 생성자 파라미터 확장 ▲▲▲

        self.dropout = nn.Dropout(p=0.5)

        # ▼▼▼ [변경] fc 입력 차원 계산 수정 ▼▼▼
        # 기존: hidden_dim 고정 (단방향이므로 은닉 상태 크기 = hidden_dim)
        # 변경: 양방향이면 순방향+역방향 은닉 상태를 이어 붙이므로 hidden_dim * 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
        # ▲▲▲ [변경] fc 입력 차원 계산 수정 ▲▲▲

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 토큰 배열을 받아 각 카테고리에 대한 예측 점수를 반환한다."""

        embedded = self.embedding(x)                   # [배치, 문장길이] → [배치, 문장길이, 임베딩차원]
        _, (hidden, _) = self.lstm(embedded)            # hidden: [num_layers * num_directions, 배치, hidden_dim]

        # ▼▼▼ [변경] 양방향 LSTM 은닉 상태 결합 처리 ▼▼▼
        # 기존: hidden[-1] 만 사용 (항상 마지막 단방향 층 은닉 상태)
        # 변경: 양방향이면 마지막 층의 순방향(-2)과 역방향(-1) 은닉 상태를 cat 으로 합침
        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [배치, hidden_dim*2]
        else:
            last_hidden = hidden[-1]                                   # [배치, hidden_dim]
        # ▲▲▲ [변경] 양방향 LSTM 은닉 상태 결합 처리 ▲▲▲

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits
