# Naver 뉴스 페이지 기사 분류기 PyCharm 프로젝트

이 프로젝트는 RNN을 활용한 자연어 처리와 네이버 기사 분류 실습을 단계적으로 다룹니다. 
전체 흐름은 딥러닝과 RNN 개념 이해, 자연어 처리 코딩 기초, 문서 자동 분류 개념, 
기사 데이터 제외어 처리, 토큰 처리, 패딩, 라벨 인코딩, RNN/LSTM 모델 구현, 학습, 평가, 
실제 기사 분류, 그리고 하이퍼파라미터 최적화로 구성됩니다.

주요 구현 내용은 다음과 같습니다.
1. 기사 문장을 정제하고 불용어를 제거합니다.
2. 단어를 정수 인덱스로 변환하는 사전을 생성합니다.
3. 서로 다른 길이의 기사 문장을 동일한 길이로 패딩합니다.
4. 문자열 카테고리를 정수 라벨로 인코딩합니다.
5. Embedding + LSTM + Dropout + Linear 구조의 문서 분류 모델을 학습합니다.
6. 학습된 모델을 파일로 저장하고 새 기사 문장을 다시 분류합니다.

## 프로젝트 구조
```
bbc_rnn_classifier_project/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ preprocess.py
│  ├─ model.py
│  ├─ train.py
│  └─ predict.py
├─ models/
├─ data/
├─ main.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 실행 방법
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python main.py
```



