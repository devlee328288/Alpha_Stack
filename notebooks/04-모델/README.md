# 04-모델

KOSPI200의 미래 5거래일 방향을 학습·예측하고 모델별 결과를 비교합니다.

주로 쓰는 사람: **오준영**

파일 이름은 `NN.주제.ipynb`입니다. 규칙은 [상위 README](../README.md)를 보세요.

## 현재 상태

- A~F 여섯 피처 조합을 Logistic Regression·RandomForest·XGBoost·LightGBM으로 비교
- 조합마다 기본·Daily Return·5Day Return·두 수익률 동시 추가를 실행
- 모델 노트북 96개와 `05.모델비교.ipynb` 24개 실행 완료
- 조합별 네 모델의 최우수 결과를 `실험/조합별 best result/`에 정리
- HF 전 종목 시장 내부 피처는 추가 실험 결과 성능 개선이 없어 A~F에는 미반영

A~F 피처 설명, 공통 검증 조건과 최신 결과는
[실험 README](실험/README.md)에 정리했습니다.
