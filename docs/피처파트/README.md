# 피처파트 — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

**피처 코드(`features/`)는 신장환 님 파트이고, 이 폴더는 그 작업을 문서로 남기는 자리입니다**
(문서화는 이동원 담당). 코드와 PR 이 정본이고, 여기 적힌 것과 코드가 다르면 **코드가 맞습니다.**
데이터 파트는 **자료로 잰 것**과 **팀이 합의한 것**만 적습니다.

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| [**version1.0**](version1.0/) ⭐ | 2026-09-05 | [피처 두 층과 중장기 수익률](version1.0/피처_두_층과_중장기_수익률.md) — `features/` 의 **원자 층 ↔ 조립 층** · 조립 층이 원자 함수를 다시 짜던 것을 이음(#107 → PR #114) · **월간 수익률**(`monthly_return` · 20거래일 · PR #116) — 주간은 `five_day_return` 이 이미 있었다 · 조합 A~F · 극단 84행 필터 판단 |

## 문서에 아직 없는 것 (이슈로만 있는 것)

| 무엇 | 어디 | 코드 |
|---|---|---|
| ③ 종목 트랙 극단수익률 필터 실측 — top_n=50 후보군과 **0건** 겹침 · `supply/stock_training_universe.py` 설계 | [#132](https://github.com/devlee328288/Alpha_Stack/issues/132) | 없음 (오준영 님 top_n 결정 대기) |
| `atr_ratio` ↔ `hv_20` 상관 **0.926** · 파생 피처 4개(`atr_ratio` · `macd_hist_atr` · `hv_regime` · `obv_slope_20`) 예정 | [#37 댓글](https://github.com/devlee328288/Alpha_Stack/issues/37) | 없음 (09-07 예정) |

코드가 들어오면 v1.1 로 옮겨 적습니다.

## 이 폴더에 무엇을 넣나

| 넣는다 | 넣지 않는다 |
|---|---|
| 피처의 **정의·시점 규칙·실측**(상관·결측·범위) | 피처 코드 자체 (→ [`features/`](../../features/)) |
| 조립 층이 원자 층을 어떻게 쓰는지 | 되돌릴 수 없는 결정 (→ [`decisions/`](../decisions/) ADR) |
| 팀 합의가 필요한 논점과 결론 | 실험 출력 원본 (→ [`notebooks/`](../../notebooks/)) |

## 다음 버전을 팔 때

```bash
cp -r docs/피처파트/version1.0 docs/피처파트/version1.1
# version1.1 을 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```
