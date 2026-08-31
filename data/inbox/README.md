# data/inbox — 팀원이 건네준 파일을 놓는 자리

이 폴더의 **파일은 커밋되지 않습니다.** `.gitignore` 가 이 README 만 남기고 전부 막습니다.

## 왜 올리지 않나

시세는 KRX 이용약관 제11조 ②가 제3자 제공을 금지하고, 뉴스 본문은 언론사 저작물이며,
이 저장소는 **PUBLIC** 입니다. 재현에 필요한 것은 파일이 아니라

- `ingest/inbox/schemas/` — 무엇이 들어와야 하는가 (규격)
- `reports/inbox/` — 무엇이 들어왔고 어떻게 판정했나 (기록)

이고, 그 둘은 커밋합니다.

## 어떻게 놓나

종류별 폴더 아래에 받은 파일을 그대로 둡니다. **고치지 말고 원본 그대로** 두세요 —
무엇을 어떻게 고쳤는지는 검사기가 기록으로 남깁니다.

```text
data/inbox/
  ohlcv_stock/    종목 시세
  ohlcv_index/    지수 시세
  news/           뉴스·텍스트
  financial/      재무·공시
  macro/          거시 지표
```

## 팀원은 어디에 올리나

팀원은 이 폴더가 아니라 **HuggingFace** 의 `inbox/<이름>/` 에 올립니다.
받는 방법·올리는 방법은 [팀원 HuggingFace 가이드](../../docs/데이터파트/version2.1/팀원_HuggingFace_가이드.md)
를 보세요. 팀장이 내려받아 이 폴더에 놓고 검사를 돌립니다.

## 다음 단계

검증·정제 엔진과 `inbox_accepted`/`inbox_quarantine` 적재는 아직 만드는 중입니다.
지금은 규격 1장(`ohlcv_stock.json`)만 서 있습니다.
