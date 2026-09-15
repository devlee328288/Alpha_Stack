# 평가파트 — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

**평가·백테스트 코드(`evaluation/` · `backtest/`)는 강민석 님 파트이고, 이 폴더는 그 작업을
문서로 남기는 자리입니다** (문서화는 이동원 담당). 코드와 PR 이 정본이고, 여기 적힌 것과
코드가 다르면 **코드가 맞습니다.** 데이터 파트는 **자료로 잰 것**과 **팀이 합의한 것**만 적습니다.

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| [**version1.4**](version1.4/) ⭐ | 2026-09-15 | 🆕 [대시보드 9페이지와 배포](version1.4/대시보드_9페이지와_배포.md) 를 main `cff6715` 로 다시 잰 판 — **Model Lab 조합이 공식 모델과 같아졌다**(MARKET C · STOCK · UNIVERSE K · PR #287 · #289) · 최고 모델은 여전히 조화평균(칸 이름만 `BEST HARMONIC MODEL` · #300) · **Backtest ↔ Cost Sens 설정 하나를 공유**하고 전략이 A~F(5거래일 락 · #301) · 논의 이슈 4건 닫힘과 지금 코드(STOCK 라벨 ±1.0% · 비용은 매수 · 매도 각각 그대로) · 수정 뒤 Space 빌드를 화면 글자로 확인 · ruff 220 · 🔴 조정 라벨이 주석(시가)과 달리 종가 · 화면 LR 0.3736 ≠ 공식 0.4542 ([변경사항](version1.4/변경사항.md)) |
| [version1.3](version1.3/) | 2026-09-14 | 🆕 [대시보드 9페이지와 배포](version1.3/대시보드_9페이지와_배포.md) — **화면 안에서 학습하는 연구 단말**(Baseline CMA-ES · Model Lab 4모델 워크포워드 · Backtest · Cost Sens · Risk · Universe · QFRS · PR #261 · #262) · 결과 캐시는 모듈 전역이라 재시작 전까지 모든 접속자 공유 · 배포 기록(Streamlit Cloud: 요구사항 파일 PR #266 · `HF_TOKEN` 이름 · CPU 제한 → HF Space Docker 로컬 검증) · 논의 이슈 5건 · ruff `dashboard/` 199 · ✏️ 오후 보강 — Frozen 제거(PR #279) · #280 BEST MODEL 이 공식 최종 모델과 다름 ([변경사항](version1.3/변경사항.md)) |
| [version1.2](version1.2/) | 2026-09-10 | [동적 기준선 · ARIMA 동반 · 백테스트 원장](version1.2/동적기준선_ARIMA동반_백테스트원장.md) — `threshold_tuning/` 재편 · Focal Loss MLP · 원장이 시가축으로(PR #218) · 🔴 라벨 축이 파일마다 다르다 ([변경사항](version1.2/변경사항.md)) |
| [version1.1](version1.1/) | 2026-09-06 | [동적 기준선 · ARIMA 동반 · 백테스트 원장](version1.1/동적기준선_ARIMA동반_백테스트원장.md) — 🆕 **ARIMA 가 워크포워드 안에서 같이 채점**(step5 · 143폴드 평균 39.85% · PR #130) · 🆕 **원장에 되먹임 칸 9개**(`signal_log` 8→17 · `trade_log` 13→22 · `predict_5d_after` 가 확률 3칸까지 · PR #131 · 칸 수는 실행으로 셈) · 🔴 **원장은 아직 HF 에 없다**(코드만) · 🔴 **한 표에 폴드 체계가 둘**(expanding 12 · gap 5 vs sliding 143 · gap 0)·라벨 지평이 둘(5일 ±1.0% vs 1일 +0.5/−0.3) → [#133](https://github.com/devlee328288/Alpha_Stack/issues/133) 회의 안건 · 시도 횟수 계측은 아직 없음(`trials.jsonl` 0바이트 · 상한이 21,450 이 아니라 코드로는 42,900) · ruff 26 → 3 ([변경사항](version1.1/변경사항.md)) |
| [version1.0](version1.0/) | 2026-09-05 | [기준선 최적화와 피처 계약](version1.0/기준선_최적화와_피처_계약.md) — `config/features.py` **제외가 아니라 포함 목록**(#84 → PR #112) · 동적 기준선 **6단계 CMA-ES 워크포워드**(PR #117 · 2,538줄) · 비대칭 계수 6개 · `balanced_accuracy` 와 #33 · HF 백테스트 결과 4종 실측 |

## 이 폴더에 무엇을 넣나

| 넣는다 | 넣지 않는다 |
|---|---|
| 워크포워드·기준선·거래비용의 **정의와 실측** | 평가 코드 자체 (→ [`evaluation/`](../../evaluation/) · [`backtest/`](../../backtest/)) |
| PR 문구와 코드가 다른 곳 — **질문 형태로** | 되돌릴 수 없는 결정 (→ [`decisions/`](../decisions/) ADR) |
| HF 에 올라간 백테스트 결과의 **모양과 축** | 실험 출력 원본 (→ [`notebooks/`](../../notebooks/)) |

## 다음 버전을 팔 때

```bash
cp -r docs/평가파트/version1.1 docs/평가파트/version1.2
# version1.2 를 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```
