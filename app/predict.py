"""저장된 모델을 불러와 새 기사 문장을 분류하는 모듈."""

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import torch

from app.config import Config
from app.model import TextLSTMClassifier

# ▼▼▼ [변경] 전처리 함수 임포트에 clean_korean_text 추가 ▼▼▼
# 기존: clean_text (영어 전용 정제 함수만 임포트)
# 변경: 예측 시에도 학습과 동일한 한국어 정제 함수(clean_korean_text)를 사용해야
#        학습/예측 전처리 불일치 문제를 방지할 수 있음
from app.preprocess import clean_korean_text, pad_sequences, texts_to_sequences
# ▲▲▲ [변경] 전처리 함수 임포트 수정 ▲▲▲


def load_artifacts(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """저장된 모델 가중치와 전처리 메타데이터를 불러온다."""

    meta_path = config.model_path.replace(".pt", "_meta.pkl")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # ▼▼▼ [변경] 모델 생성 시 num_layers, bidirectional 파라미터 전달 ▼▼▼
    # 기존: TextLSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
    # 변경: 저장 당시 config 에 기록된 num_layers, bidirectional 값을 그대로 사용해
    #        학습 구조와 완전히 동일한 모델을 복원
    saved_config: Config = metadata["config"]
    model = TextLSTMClassifier(
        vocab_size=len(metadata["vocab"]),
        embed_dim=saved_config.embed_dim,
        hidden_dim=saved_config.hidden_dim,
        num_classes=len(metadata["label_to_id"]),
        num_layers=saved_config.num_layers,
        bidirectional=saved_config.bidirectional,
    )
    # ▲▲▲ [변경] 모델 생성 파라미터 확장 ▲▲▲

    model.load_state_dict(torch.load(config.model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, metadata


def predict_text(text: str, model: TextLSTMClassifier, metadata: Dict[str, object], config: Config) -> str:
    """새 기사 한 문장을 입력받아 예측된 네이버 뉴스 카테고리명을 반환한다."""

    # ▼▼▼ [변경] 정제 함수 교체 ▼▼▼
    # 기존: clean_text(text) → 영어 전용, 한글 제거
    # 변경: clean_korean_text(text) → 학습 시 전처리와 동일한 한국어 정제 적용
    cleaned = clean_korean_text(text)
    # ▲▲▲ [변경] 정제 함수 교체 ▲▲▲

    sequence = texts_to_sequences([cleaned], metadata["vocab"])[0]
    padded = pad_sequences([sequence], config.max_len)
    x = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        pred_id = int(torch.argmax(logits, dim=1).item())
    return metadata["id_to_label"][pred_id]
