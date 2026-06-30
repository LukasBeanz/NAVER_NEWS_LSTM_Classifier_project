"""데이터 전처리, 모델 학습, 평가, 저장을 수행하는 모듈."""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ▼▼▼ [추가] 시각화 라이브러리 임포트 ▼▼▼
# 기존: 시각화 없음
# 변경: matplotlib 으로 학습 Loss/Accuracy 곡선과 혼동행렬(Confusion Matrix)을 PNG 로 저장
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경(서버)에서도 그래프 저장 가능하도록 백엔드 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# ▲▲▲ [추가] 시각화 라이브러리 임포트 ▲▲▲

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier

# ▼▼▼ [변경] 전처리 함수 임포트에 clean_korean_text 추가 ▼▼▼
# 기존: clean_text (영어 전용 정제 함수만 임포트)
# 변경: clean_korean_text (한국어 정제 함수) 추가 임포트
from app.preprocess import build_vocab, clean_korean_text, encode_labels, pad_sequences, texts_to_sequences
# ▲▲▲ [변경] 전처리 함수 임포트 수정 ▲▲▲


def set_seed(seed: int) -> None:
    """학습 결과가 최대한 동일하게 재현되도록 난수를 고정한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ▼▼▼ [추가] 한국어 폰트 설정 함수 ▼▼▼
# 기존: 한국어 폰트 설정 없음 (그래프에 한글이 깨짐)
# 변경: Windows/Mac/Linux 환경별로 사용 가능한 한글 폰트를 탐색해 matplotlib 에 적용
def _set_korean_font() -> None:
    """matplotlib 한글 깨짐 방지를 위해 시스템 한글 폰트를 설정한다."""
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
# ▲▲▲ [추가] 한국어 폰트 설정 함수 ▲▲▲


# ▼▼▼ [추가] 시각화 저장 함수 ▼▼▼
# 기존: 시각화 없음
# 변경: epoch 별 train_losses, train_accs, val_accs 와 혼동행렬을 2×2 서브플롯으로 저장
def _save_training_plot(
    train_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    all_targets: List[int],
    all_preds: List[int],
    id_to_label: Dict[int, str],
    plot_path: str,
) -> None:
    """학습 곡선(Loss/Accuracy)과 혼동행렬을 PNG 파일로 저장한다."""
    _set_korean_font()
    epochs = range(1, len(train_losses) + 1)
    labels = [id_to_label[i] for i in sorted(id_to_label)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("네이버 뉴스 LSTM 학습 결과", fontsize=14)

    # 서브플롯 1: 학습 손실 곡선
    axes[0].plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    axes[0].set_title("학습 손실 (Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # 서브플롯 2: 학습/검증 정확도 곡선
    axes[1].plot(epochs, train_accs, "g-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, val_accs, "r-s", markersize=3, label="Val Acc")
    axes[1].set_title("정확도 (Accuracy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    # 서브플롯 3: 혼동행렬 (Confusion Matrix)
    cm = confusion_matrix(all_targets, all_preds)
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_title("혼동행렬 (Confusion Matrix)")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=15, fontsize=8)
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel("예측 라벨")
    axes[2].set_ylabel("실제 라벨")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=axes[2])

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"시각화 저장 완료: {plot_path}")
# ▲▲▲ [추가] 시각화 저장 함수 ▲▲▲


def train_model(config: Config) -> Tuple[TextLSTMClassifier, Dict[str, object]]:
    """네이버 뉴스 기사 제목 데이터를 사용해 LSTM 문서 분류 모델을 학습한다."""

    set_seed(config.random_state)
    raw_texts, labels = load_sample_data()

    # ▼▼▼ [변경] 전처리 함수 교체 ▼▼▼
    # 기존: clean_text(text) → 영어 전용, 한글 문자를 모두 제거해 버려 한국어 불가
    # 변경: clean_korean_text(text) → 한글 보존, 한국어 불용어 제거
    cleaned_texts = [clean_korean_text(text) for text in raw_texts]
    # ▲▲▲ [변경] 전처리 함수 교체 ▲▲▲

    vocab = build_vocab(cleaned_texts, config.max_vocab)
    sequences = texts_to_sequences(cleaned_texts, vocab)
    x = pad_sequences(sequences, config.max_len)
    y, label_to_id, id_to_label = encode_labels(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # ▼▼▼ [변경] 모델 생성 시 num_layers, bidirectional 파라미터 전달 ▼▼▼
    # 기존: TextLSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
    # 변경: num_layers, bidirectional 추가 → 양방향 2층 LSTM 구성
    model = TextLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_classes=len(label_to_id),
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
    )
    # ▲▲▲ [변경] 모델 생성 파라미터 확장 ▲▲▲

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # ▼▼▼ [추가] 에폭별 지표 기록 리스트 ▼▼▼
    # 기존: 손실만 출력, 정확도 기록 없음
    # 변경: train_losses, train_accs, val_accs 를 epoch 별로 누적해 시각화에 사용
    train_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    # ▲▲▲ [추가] 에폭별 지표 기록 리스트 ▲▲▲

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds_ep: List[int] = []
        train_targets_ep: List[int] = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # ▼▼▼ [추가] 학습 배치 예측값 수집 (epoch 단위 train accuracy 계산용) ▼▼▼
            train_preds_ep.extend(torch.argmax(logits, dim=1).tolist())
            train_targets_ep.extend(batch_y.tolist())
            # ▲▲▲ [추가] 학습 배치 예측값 수집 ▲▲▲

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_targets_ep, train_preds_ep)
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # ▼▼▼ [추가] epoch 단위 검증 정확도 계산 ▼▼▼
        # 기존: 전체 학습 완료 후 1회 평가
        # 변경: 매 epoch 마다 검증 정확도도 측정해 과적합 여부를 실시간으로 확인
        model.eval()
        ep_preds: List[int] = []
        ep_targets: List[int] = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                ep_preds.extend(torch.argmax(logits, dim=1).tolist())
                ep_targets.extend(batch_y.tolist())
        val_acc = accuracy_score(ep_targets, ep_preds)
        val_accs.append(val_acc)
        # ▲▲▲ [추가] epoch 단위 검증 정확도 계산 ▲▲▲

        # ▼▼▼ [변경] 출력 포맷 확장 ▼▼▼
        # 기존: "Epoch XX/YY - loss: Z.ZZZZ"
        # 변경: train_acc, val_acc 도 함께 출력해 학습 상태를 한눈에 파악
        print(
            f"Epoch {epoch:02d}/{config.epochs}"
            f" - loss: {avg_loss:.4f}"
            f" - train_acc: {train_acc:.4f}"
            f" - val_acc: {val_acc:.4f}"
        )
        # ▲▲▲ [변경] 출력 포맷 확장 ▲▲▲

    # 최종 평가
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n최종 평가 정확도: {accuracy:.4f}")

    # ▼▼▼ [추가] 상세 평가지표 출력 ▼▼▼
    # 기존: accuracy_score + classification_report 만 출력
    # 변경: 정밀도/재현율/F1 포함 classification_report 출력 (동일 구조 유지, 한국어 라벨 적용)
    print("\n[분류 리포트]")
    print(
        classification_report(
            all_targets,
            all_preds,
            target_names=[id_to_label[i] for i in range(len(id_to_label))],
            zero_division=0,
        )
    )
    # ▲▲▲ [추가] 상세 평가지표 출력 ▲▲▲

    # 모델 저장
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    with open(config.model_path.replace(".pt", "_meta.pkl"), "wb") as f:
        pickle.dump(
            {"vocab": vocab, "label_to_id": label_to_id, "id_to_label": id_to_label, "config": config},
            f,
        )

    # ▼▼▼ [추가] 학습 결과 시각화 파일 저장 ▼▼▼
    # 기존: 시각화 없음
    # 변경: Loss 곡선, Accuracy 곡선, 혼동행렬을 PNG 파일로 저장
    _save_training_plot(
        train_losses, train_accs, val_accs,
        all_targets, all_preds, id_to_label,
        config.plot_path,
    )
    # ▲▲▲ [추가] 학습 결과 시각화 파일 저장 ▲▲▲

    metadata = {
        "vocab": vocab,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "accuracy": accuracy,
        # ▼▼▼ [추가] 메타데이터에 학습 이력 포함 ▼▼▼
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        # ▲▲▲ [추가] 메타데이터에 학습 이력 포함 ▲▲▲
    }
    return model, metadata
