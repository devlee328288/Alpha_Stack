# 평가 파트 — 동적 기준선 · ARIMA 동반 · 백테스트 원장 (v1.1)

> 2026-09-06 · 정리: 이동원(문서) · **작업: 강민석**
> 근거: PR [#112](https://github.com/devlee328288/Alpha_Stack/pull/112) ·
> [#117](https://github.com/devlee328288/Alpha_Stack/pull/117) ·
> [#130](https://github.com/devlee328288/Alpha_Stack/pull/130) ·
> [#131](https://github.com/devlee328288/Alpha_Stack/pull/131) ·
> 이슈 [#84](https://github.com/devlee328288/Alpha_Stack/issues/84) ·
> [#110](https://github.com/devlee328288/Alpha_Stack/issues/110) ·
> [#120](https://github.com/devlee328288/Alpha_Stack/issues/120) ·
> [#127](https://github.com/devlee328288/Alpha_Stack/issues/127) ·
> [#133](https://github.com/devlee328288/Alpha_Stack/issues/133)
> 실측 기준 커밋: `main` **`0bf504e`** (PR #131 머지 직후)

> 📌 **이 문서는 강민석 님이 하신 일을 제가 옮겨 적은 것입니다.** 코드와 PR 이 정본이고,
> 여기 적힌 것과 코드가 다르면 **코드가 맞습니다.** 빠진 것이나 틀린 것을 발견하시면
> 알려 주시거나 직접 고쳐 주세요. v1.0 에서 파일 이름이 바뀌었습니다 —
> 다루는 것이 둘에서 넷으로 늘어 제목을 내용에 맞췄습니다([변경사항](변경사항.md)).

---

## 0. 한 줄로

v1.0 의 둘(① 피처 계약 상수 · ② 동적 기준선 6단계)에 둘이 더 들어왔습니다.
**③ ARIMA 기준선이 워크포워드 안에서 같이 평가되고**(#130), **④ 백테스트 원장에
되먹임용 칸 9개가 뚫렸습니다**(#131). 🔴 다만 **원장은 아직 HF 에 없고**(코드만 머지),
기준선 표 하나에 **폴드 체계가 둘·라벨 지평이 둘** 섞여 있어 [#133](https://github.com/devlee328288/Alpha_Stack/issues/133) 에서 회의 안건으로 올렸습니다.

---

## 1. `config/features.py` — 피처 계약을 상수로 뽑았다 (#112)

v1.0 과 같습니다. 요지만 남깁니다.

평가·백테스트가 `X` 를 **제외 목록**으로 만들던 것을 **포함 목록** `FEATURE_COLUMNS`(22칸)으로
바꿨습니다. 배포본에 칸이 늘어도 조용히 딸려 들어가지 않습니다. 세 곳이 씁니다 —
`evaluation/evaluation_backtest.py:20` · `evaluation/evaluation_risk.py:13` · `backtest/collector_5days.py:24`.
반출 쪽 `scripts/export_team_dataset.py` 의 `FEATURE_COLUMNS` 와 **같은 목록·같은 순서**입니다.

---

## 2. 동적 기준선 6단계 파이프라인 (#117)

`evaluation/threshold_tuning/` · 설계 문서는 같은 폴더의 `threshold_tuning.md`.

### 수식 — 하이브리드 기준선

```
중립 기준선  Base                                   (SMA(종가, 20) 고정 · 튜닝 안 함)
상승 기준선  Base + (α_up   × 변동성) + (β_up   × 거래량조정항)
하락 기준선  Base − (α_down × 변동성) − (β_down × 거래량조정항)
```

튜닝 대상은 `α_up · α_down · β_up · β_down` 과 변동성·거래량 측정 기간까지 **6개**입니다.
🔴 **기준선 3종(`always_up` · `majority_class` · `previous_direction`)은 튜닝 대상이 아닙니다** —
파라미터가 없습니다. 이름이 비슷해 겹쳐 보일 뿐입니다(#127).

### 단계

| 단계 | 파일 | 무엇 |
|---|---|---|
| 1 | `step1_core_features.py` | 코어 피처 — Base(SMA20) · 변동성 · 거래량 조정항 · HF `small/features_labels_kospi200_dev.csv` 적재 |
| 2 | `step2_optimize.py` | 4파라미터 CMA-ES 워크포워드 |
| 3 | `step3_objective_optimize.py` | 목적 함수 — `Sharpe − λ×MDD` |
| 4 | `step4_walkforward_official.py` | 정식 워크포워드 (학습 구간 이동) |
| 5 | `step5_optimize_6params.py` | 6파라미터 전체 탐색 · 🆕 **폴드마다 ARIMA 를 새로 적합해 같이 채점** (§3) |
| 6 | `step6_live_signal.py` | 실전 신호 생성 · 클래스 비율 · `balanced_accuracy` |
| — | `run_full_pipeline.py` | 1~6 을 한 번에 · 🆕 전체 정확도 출력 |

### 🔴 2.1 step5 의 워크포워드는 합의 체계와 다릅니다 — 코드 실측

| | 합의 체계 (계획서 v4.1 · `models/experiment.py` · `evaluation/walk_forward.py`) | `step5` |
|---|---|---|
| 분할 | **expanding** | **sliding** |
| 폴드 수 | **12** (`walk_forward.py:43`) | **143** (`step5:251` · (3,553−504−63+1)÷21 올림) |
| 학습 창 | 최소 750 일, 이후 확장 | 고정 **504** 일 (`:223`) |
| 검증 창 | 60 일 | **63** 일 (`:224` · 3×21) |
| 이동 폭 | 검증창만큼 (겹침 없음) | **21** 일 (`:225`) — 검증창이 세 번씩 겹침 |
| gap | **5** (= label_horizon · 5 미만이면 `LeakageError`) | **0** (`:253` `val_end = train_end + val_days`) |
| 검증 커버리지 | 720 / 3,553 = **20.3%** | 약 3,045 시점(85.7%) · 겹침 포함 9,009 |

어느 쪽이 맞다는 것이 아니라 **한 표 안에 둘이 섞이면 안 된다**는 것입니다.
[#133 §5](https://github.com/devlee328288/Alpha_Stack/issues/133) 에서 어느 체계로 맞출지
회의에서 정합니다. 계획서 ⑧ 검증 절과 `models/experiment.py` 가 이미 expanding 12폴드라
그쪽으로 맞추는 편이 문서·모델과 한 줄로 섭니다.

### `balanced_accuracy` 와 이슈 #33

`step6` 이 `balanced_accuracy_score` 와 클래스별 비율을 함께 봅니다. `hit_rate` 가 중립을
분모에서 빼는 문제(#33)를 우회하는 방향입니다. 다만 **`evaluation/metrics.py:112` 의
`hit_rate` 자체는 아직 그대로**입니다.

---

## 3. 🆕 ARIMA 동반 (#130) — "원래 쓰던 방법보다 나은가"에 답하는 행

계획서 F-14 · 분석 방법 ⑧ *"기준선 3종 · **ARIMA 동반**"*. 기준선 3종은 전부 *머리를 안 쓰는*
상대라, 이걸 이겨도 *"원래 쓰던 방법보다 나은가"* 에는 답이 안 됩니다. ARIMA 가 그 상대입니다.

### 어떻게 들어갔나 (`step5_optimize_6params.py`)

| 줄 | 무엇 |
|---|---|
| `:15` | `from timeseries.models import fit_best` (파일 최상단으로 이동 · `sys.path` 를 루트로) |
| `:332` | 학습 창의 **1거래일 수익률** `df_train['close'].pct_change()` |
| `:337` | `fit_best(train_ret, max_p=3, max_q=3)` — 폴드마다 새로 적합 (미래 정보 없음 ✅) |
| `:352` | 검증 창 길이만큼 **하루씩 재귀 예측** (`phi` · `const` · `levels` 로 수동 계산) |
| `:371-373` | 예측 수익률을 **±1.0%** 로 상승/중립/하락 |
| `:319` | 비교 대상 `y_true` = `label` 칸 (5거래일 · ±1.0%) |
| `:455-458` | 폴드별 정확도의 **평균** 출력 |

### 결과 (강민석 님 로그 · 제가 재현하지 않았습니다)

| 예측기 | 방향 적중률 | 근거 |
|---|---:|---|
| `always_up` | 33.58% | 개발 구간 라벨 분포(상승) — 폴드 없음 |
| `majority_class` | 38.50% | 개발 구간 라벨 분포(중립) — 폴드 없음 |
| `previous_direction` | 84.08% | 라벨 겹침(4/5) · 실전 불가 · **상한선 참고용** |
| **`ARIMA`** | **39.85%** | step5 · **143폴드 평균** |
| `threshold_pipeline` | 41.08% | `run_full_pipeline.py:247` 전체 정확도 |

### 🔴 여쭙는 것 둘 (#133 §4 · 회의 전 확인)

1. **ARIMA 의 예측 지평** — 1일 수익률을 예측해 5일 라벨과 비교하는 것으로 읽힙니다.
   코스피 200 개발 구간 실측으로 **1일 σ 1.106% · 5일 σ 2.456%** 이고 하루 수익률이 ±1.0% 를
   넘는 날은 **28.5%** 뿐이라(5일은 62.4%), 예측값은 대부분 중립으로 찍힙니다.
   39.85% 가 다수 클래스 38.50% 옆에 붙어 있는 것이 그 모습일 수 있습니다.
2. **`threshold_pipeline` 41.08% 의 라벨** — `run_full_pipeline.py:69-70` 이
   `close.pct_change()` (**1거래일**) 에 **+0.5% / −0.3%** 를 대서 `y_true` 를 만듭니다.
   표의 1·2행(5거래일 · ±1.0%)과 다른 문제에 대한 숫자로 읽힙니다.

---

## 4. 🆕 폴드 커버리지 · 시도 횟수 계측 (#127 → #130)

계획서 v4.0 「용어와 산출물 사전」이 정의한 셋입니다. 지금 상태를 적습니다.

| 용어 | 무엇을 만들면 끝인가 | 지금 |
|---|---|---|
| ARIMA 동반 | 성능표에 ARIMA 행 하나 | ✅ 행은 생겼다 (§3) · 🔴 지평·폴드 체계는 #133 |
| 폴드 커버리지 | "검증창이 개발 구간의 몇 %를 덮었나" 한 줄 | ✅ **20.3%** = 720 / 3,553 (expanding 12폴드 · 검증 60 · gap 5 · 09-05 실측) |
| 시도 횟수 계측 | `reports/trials.jsonl` 행 수 == `fit` 호출 수 | 🔴 **아직 없습니다** — 아래 |

### 🔴 시도 횟수 — 코드 실측 (09-06 · `0bf504e`)

- `reports/trials.jsonl` 은 **0 바이트** 빈 파일입니다 (PR #130 이 만들었습니다).
- PR 본문의 `log_trial` 함수는 **저장소 `.py` 어디에도 없습니다** (전수 `grep`). 로컬에만 있을 수 있습니다.
- PR 본문의 상한 **21,450회 = 143폴드 × 150** 인데, 코드는 `step5:224` 기본값 **`max_evals=300`** ·
  `run_full_pipeline.py:81` 정밀 모드도 **300** → `:279` `maxfevals` 로 들어갑니다.
  코드대로면 상한은 **143 × 300 = 42,900** 입니다. 150 이 어디서 온 값인지 확인이 필요합니다.
- `evaluation/evaluation_risk.py:529` 의 `n_trials=50` (Deflated Sharpe 가정값)은 그대로입니다.

CMA-ES 는 수렴하면 `maxfevals` 전에 멈추므로(`:288` `while not es.stop()`), 정확한 횟수는
그 루프 안에서 한 번 부를 때마다 한 줄씩 적어야 나옵니다. PR #130 이 "재실행 시 계측" 으로
남겼습니다.

---

## 5. 🆕 백테스트 원장 — 되먹임용 칸 9개 (#110 → #131)

강사님 09-04 응답 *"백테스팅 결과를 데이터 형식으로 받아 학습·예측에 되먹이는 설계"* 의 첫 단계입니다.
데이터 파트가 #110 에서 칸 다섯을 여쭈었고, 강민석 님이 **아홉 칸**으로 넓혀 넣으셨습니다.

### 무엇이 바뀌었나 (`backtest/backtest_strategies.py`)

| 무엇 | 전 | 후 |
|---|---|---|
| `predict_5d_after()` 반환 | `"상승"` 문자열 하나 | **`(라벨, {"p_up", "p_flat", "p_down"})`** — 지금은 상수 0.33/0.34/0.33 (랜덤 예측) |
| 구버전 예측 함수 | — | 문자열만 돌려줘도 기본 확률로 채워 동작 (호환) |
| `run_backtest()` 인자 | — | `model_id="baseline-v0"` · `run_id=None`(자동: `run_YYYYMMDD_HHMMSS_xxxxxxxx`) |
| `realized_return_5d` | 없음 | `close[t+5] / close[t] − 1` · 마지막 5일은 NaN (인덱스 초과 방지) |

### 원장 칸 — 실측 (`0bf504e`)

| 원장 | 기존 | 🆕 추가 9칸 | 합계 |
|---|---:|---|---:|
| `signal_log` (hold 포함 신호 전량) | 8 | `code` · `p_up` · `p_flat` · `p_down` · `realized_return_5d` · `model_id` · `model_rev` · `run_id` · `cost_rate` | **17** |
| `trade_log` (체결만) | 14 | 같은 9칸 | **23** |

`code` 는 지금 상수 `"KOSPI200"` 입니다. 종목으로 넓힐 때 그대로 씁니다. `model_rev` 는 `"v0"` 고정.

### HF 업로드 — 코드는 있고, 🔴 **실행은 아직입니다**

`backtest/run_cost_sensitivity.py` 가 모든 실행의 `signal_log`·`trade_log` 를 모아
(`strategy` · `cost_key` 칸을 붙여) 두 저장소로 `push_to_hub` 합니다.

| 저장소 | 무엇 | 09-06 실측 |
|---|---|---|
| `qurious-quant/alphastack-backtest-execution-log` | 신호 전량 (hold 포함) | 🔴 **없음** |
| `qurious-quant/alphastack-backtest-trade-log` | 체결만 | 🔴 **없음** |

`HfApi().list_datasets(author="qurious-quant")` 결과는 여전히 **6개**(krx-dev · dart · 백테스트 요약 4종)입니다.
`run_cost_sensitivity.py` 를 한 번 돌리면 올라갑니다. #110 에 요청해 두었습니다.

### 되먹임 피처는 2차 (합의)

#110 에서 제안한 대로 **1차는 원장 반출까지**입니다. 지금 예측이 랜덤이라 되먹여도 배울 것이
없고, "최근 20거래일 몇 번 맞았나" 같은 피처는 `known_at` 을 체결일 다음 거래일로 잡지 않으면
누수입니다. 강민석 님도 같은 판단으로 뺐습니다(PR #131 ❌ 목록).

### 데이터 파트가 이어서 하는 것

| 무엇 | 언제 |
|---|---|
| 원장 **계약 테스트** — 칸 9개 존재 · 확률 합 1 · `execution_date ≥ prediction_date` · 홀드아웃(20240901~) 행 0 · `realized_return_5d` 가 원가격 5일 수익률과 일치 | 이번 세션 (`tests/`) |
| 원장 ↔ `daily_price`·`adj_close`·라벨 **조인 정문** (`supply/`) · 반출 게이트 등록 | 원장이 HF 에 올라온 뒤 |

---

## 6. 데이터를 어디서 읽나

v1.0 과 같습니다. 전부 HF `qurious-quant/alphastack-krx-dev` 의 지수 파일 하나 —
`small/features_labels_kospi200_dev.csv` (3,553행 × 37칸). ⚠️ 종목(`stocks30`)으로 바꾸실 때
`pct_change()` 10곳은 `groupby("code")` 가 필요하고, 숫자 아닌 칸 5개는 `config/features.py` 가 막습니다.

---

## 7. 남은 것

- [ ] **#133 회의 결정 4가지** — 표를 한 체계로 다시 잴지 · ARIMA 지평 · `threshold_pipeline` 라벨 · step5 gap
- [ ] **원장 HF 업로드 실행** — `run_cost_sensitivity.py` 한 번 (#110)
- [ ] `trials.jsonl` 계측 — `log_trial` 을 저장소에 넣고 `es.stop()` 루프 안에서 호출 · `n_trials=50` 치환
- [ ] `ruff` — 09-05 에 0건이었다가 PR #130·#131 로 **26건**. 이 문서와 같은 PR 에서 공백·import
      정렬·f-string·EOF 개행 **23건은 기계적으로 정리**했고(동작 변화 없음 · `sys.path` 삽입은 그대로
      import 앞), **3건이 남았습니다** — 코드 줄을 고쳐야 해서 강민석 님 몫입니다:
      `backtest_strategies.py:128` E501(105자) · `step5:15` F401 + `:332` F811(`fit_best` 를 위와 루프 안에서 두 번 import — 안쪽 것을 지우면 둘 다 사라집니다)
- [ ] `evaluation/evaluation_risk.py:509` 의 `change_rate / 100` — KRX 소수 2자리 반올림 값 (#84)
- [ ] `metrics.py::hit_rate` 가 중립을 분모에서 빼는 문제 (#33)

---

## 8. HF 에 올라간 백테스트 결과 (2026-09-06 실측 · 전부 public)

| 저장소 | 모양 | 축 | 마지막 갱신(UTC) |
|---|---|---|---|
| `alphastack-backtest-results` | 3 × 10 | 전략 | 09-05 14:48 |
| `alphastack-backtest-kospi200` | 403 × 4 | 날짜 (A/B/C 일별 수익률) | 09-05 14:48 |
| `alphastack-cost-sensitivity` | 12 × 14 | 전략 × 비용 | 09-05 14:47 |
| `alphastack-breakeven-cost` | 3 × 2 | 전략 | 09-05 14:47 |
| `alphastack-backtest-execution-log` | — | (날짜 × 전략 × 비용) · 17칸 예정 | 🔴 미업로드 |
| `alphastack-backtest-trade-log` | — | 체결 · 23칸 예정 | 🔴 미업로드 |

발표·비교에는 위 4종이 맞고, **학습에 되먹이는 축(종목·날짜·예측확률)은 아래 2종**에 있습니다.
