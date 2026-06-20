## Day1
### 1. 시계열 전처리 (09:00~12:00)
* CWRU Bearing 진동 데이터 처리
* 시계열 로드, 세그멘테이션, FFT, 정규화
* 강의
    * 회전기계 진동 신호 특성
    * FFT
    * 시계열 윈도잉
* 실습
    * .mat 로드
    * 윈도우 슬라이싱
    * FFT 스펙트럼
    * 정규화 파이프라인
* 산출물
    * 전처리 완료 시계열 데이터셋
---
### 2. 1D-CNN 학습
* PyTorch 1D-CNN으로 베어링 4클래스 분류 베이스라인
* 강의
    * 1D-CNN 구조
    * 진동 신호 특화 layer 설계
* 실습
    * 1D-CNN 학습
    * Accuracy 98% 벤치
    * Confusion Matrix
* 산출물
    * 1D-CNN 모델 '.pt' + 평가 리포트
---
### 3. CNN-LSTM, 실설비 매핑
* CNN-LSTM 고도화, Nature2025 시멘트 실설비 사례 매핑 분석
* 강의
    * CNN-LSTM 구조
    * 시멘트 회전기계 (분쇄기, 킬른, 송풍기) 적용 시나리오
* 실습
    * CNN-LSTM 학습 99%
    * 벤치, 롤러프레스 IR, 원심팬, 배기팬 매핑
* 산출물
    * CNN-LSTM 모델 + 사내 적용 시나리오 문서

## Day2
### 1. Auto ML
* PyCaret으로 10+ 모델 자동 학습/비교
* 강의
    * PyCaret 구조, setup, compare, tune 흐름
* 실습
    * UCI Concrete, 부서 데이터로 자동 비교, Top3 선정

### 2. 모델 해석과 개선
* SHAP Summary Plot, 변수 중요도 Top5 시각화
* Optuna 하이퍼 튜닝
* 강의
    * SHAP, LIME 원리
    * Optuna 사용법
* 실습
    * SHAP Summary, Force Plot
    * Optuna 튜닝

### 3. 로컬 모델 앱
* 선정 모델을 Gradio, Streamlit 인터페이스로 감싸 현업 테스트
* 강의
    * Gradio, Streamlit 차이
    * UI 패턴
* 실습
    * 로컬 구동 가능한 웹 앱 데모 UI 구성

## Day3
### 1. FastAPI와 Swagger
* FastAPI를 이용해 모델을 REST API로 노출, Swagger UI 자동 생성
* 강의
    * FastAPI란?
    * Pydantic 입력 검증과 비동기 흐름
* 실습
    * Colab에 FastAPI 코드 작성
    * Swagger 테스트
### 2. ngrok을 활용한 URL 공개
* Colab에서 ngrok 터널로 외부 공개 URL 발급