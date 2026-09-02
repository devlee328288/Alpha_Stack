# pipelines/ — 재현 가능한 단일 진입점

> 필수범위 ④ **"한 명령으로 같은 결과"** 가 여기서 닫힙니다.
> 🟢 **수집(`ingest`)은 구현됐습니다** (2026-09-02). 나머지는 아직 계획입니다.

---

## 지금 쓸 수 있는 것

```bash
python -m pipelines.ingest              # 시세 → 지수 → 재무 → 거시
python -m pipelines.ingest --only macro # 골라서
python -m pipelines.ingest --dry-run    # 무엇을 할지만 보고 받지는 않는다
python -m pipelines.ingest --status     # 최근 실행 기록
```

실측 2026-09-02: 전체 4단계 **12초** (이미 받은 것은 건너뜁니다).

진행 상황은 `ingest_run` · `ingest_run_stage` 에 **시작할 때부터** 남습니다.
끝나고 한꺼번에 쓰면 중간에 죽었을 때 아무 흔적도 없어서 *"애초에 안 돌았다"* 와
구별되지 않기 때문입니다. 팀 대시보드는 이 두 표를 폴링합니다.

자세한 것: [기능명세 v1.3](../docs/기능명세/version1.3/거시수집과_수집파이프라인.md)

---

## 왜 필요한가

노트북에서 셀을 순서대로 눌러 나온 결과는 재현되지 않습니다.

- 누가 어떤 셀을 몇 번 돌렸는지 남지 않습니다
- 중간에 변수를 손으로 고치면 그 사실이 사라집니다
- 다른 사람이 같은 노트북을 받아도 같은 숫자가 나오지 않습니다

발표에서 "이 결과가 어떻게 나왔나요"라는 질문에 답할 수 있어야 합니다.
그 답이 **명령어 한 줄**이어야 합니다.

---

## 만들 것 (초안)

```bash
# 전체 파이프라인 — 수집부터 성과 보고까지
python -m pipelines.run --config configs/baseline.yaml

# 단계별로도 돌 수 있게
python -m pipelines.ingest      # 시세 수집·갱신
python -m pipelines.features    # 피처 표 생성
python -m pipelines.train       # 폴드별 학습
python -m pipelines.evaluate    # 성과 검증 + 기준선 비교
```

### 설정은 파일로, 인자로 흘리지 않는다

```yaml
# configs/baseline.yaml (예시 — 확정 아님)
universe: core            # core(350) | index(KOSPI200) | custom
target: KOSPI200
label:
  horizon: 1              # 며칠 뒤를 맞히나
  threshold: 0.0          # 보합 임계값 (0 이면 2분류)
  price: close            # close | open
features: [ma, rsi, macd, bollinger, volume]
split:
  method: expanding       # expanding | rolling
  n_folds: 12
  min_train: 120
  gap: 0                  # ⚠️ label.horizon 이 1보다 크면 반드시 맞춘다
models: [random_forest, xgboost, lightgbm]
cost:
  round_trip: 0.003       # ⚠️ 가정치 — docs/회의안건/2026-09-01-킥오프.md B-2
seed: 42
```

**설정 파일을 결과와 함께 저장합니다.** 그래야 "이 숫자가 어떤 조건에서 나왔나"를
나중에 답할 수 있습니다.

---

## 재현성을 위해 지킬 것

| 항목 | 규칙 |
|---|---|
| 난수 | `seed` 를 설정에 두고 모든 모델에 전달 |
| 버전 | `pyproject.toml` 이 정확한 버전을 고정 (실측으로 정한 조합) |
| 데이터 | 어느 시점 스냅샷인지 결과에 기록 (`ohlcv_sync_log` 참조) |
| 설정 | 결과 폴더에 사용한 config 를 복사 |
| 출력 | `artifacts/` 아래 · **git 이 추적하지 않는다** |

> ⚠️ **학습 결과물(pkl)을 팀원끼리 주고받지 않습니다.** 그러기 시작하면
> "누구 모델이 맞나"를 아무도 답할 수 없게 됩니다. 파이프라인을 다시 돌려서
> 같은 숫자가 나오는 것이 곧 재현입니다.

---

## 막고 있는 것

이 폴더를 채우려면 먼저 정해져야 합니다.

- 예측 대상과 레이블 정의 → [회의안건 A-2](../docs/회의안건/2026-09-01-킥오프.md)
- 지표 목록 → [회의안건 B-1](../docs/회의안건/2026-09-01-킥오프.md)
- 진입점 형태 (`python -m` · `invoke` · `make`)
