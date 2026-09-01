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

## 어떻게 들이나

```bash
python scripts/check_inbox.py            # 이 폴더 + HuggingFace inbox/ 를 훑고 새 것만 들인다
python scripts/check_inbox.py --dry-run  # 검사만 하고 DB 에는 안 담는다
python scripts/check_inbox.py --force    # 규격을 고친 뒤 다시 검사한다
```

같은 파일을 두 번 들이지 않습니다. 판단은 이름이 아니라 **내용 지문(SHA-256)** 이라,
이름을 바꿔 다시 놓아도 건너뜁니다.

폴더 이름이 종류를 알려 주지만, 종류 폴더 없이 두어도 됩니다 — 그때는 규격 5장에 대 보고
잽니다. **애매하면 정하지 않고 물어봅니다.**

`_hf/` 는 HuggingFace 에서 내려받은 것이 쌓이는 자리입니다. 함께 `.gitignore` 됩니다.

## 판정은 어디에 남나

| 어디 | 무엇 | 커밋하나 |
|---|---|---|
| `reports/inbox/<날짜>/*.md` | 사람이 읽는 판정 — 무엇이 왜 안 들어갔나 | ✅ |
| `reports/inbox/<날짜>/*.json` | 기계가 읽는 판정 — 전량 집계 + 표본 20건 | ✅ |
| `krx_cache.db` 의 `inbox_accepted` | 합격한 행 | ❌ (DB 는 로컬) |
| `krx_cache.db` 의 `inbox_quarantine` | 격리된 행 **+ 원본** | ❌ |

격리된 행은 원본까지 함께 담습니다. 고쳐서 다시 넣으려면 우리가 정제하기 전 값이 필요하니까요.
