"""공시 제목에 감성 모델을 돌려 `text_signal` 을 채운다.

    python scripts/score_text_signal.py --status     # 받지 않고 현황만
    python scripts/score_text_signal.py --limit 200  # 200개만 (시험 삼아)
    python scripts/score_text_signal.py              # 아직 안 매긴 것 전부

## 🔴 행마다 매기지 않는다 — 고유 제목마다 매긴다

`dart_disclosure` 1,555,556행의 **고유 제목은 18,600개(1.2%)** 다(실측 2026-09-04).
공시 제목이 정형 문구라 같은 문장이 평균 83번 반복된다. 행마다 추론하면 같은 문장을
83번씩 다시 읽는 셈이고 시간이 83배 든다.

    행마다      1,555,556건 ÷ 98.8건/초 = 약 4.4시간
    고유 제목만    18,600건 ÷ 98.8건/초 = 약 3.1분

## 비용은 0원이다

모델을 **내려받아 로컬로** 돌린다. HuggingFace 원격 추론은 건당 $0.01~0.03 이라
18,600건이면 상한($1.50)을 훌쩍 넘는다. 로컬은 0원이고, 이 규모에서는 GPU 도
필요 없다 — CPU 로 98.8건/초가 나온다(실측).

## 무엇을 남기나

`text_signal` 에 `(text_sha, model_id)` 로 담는다. **모델 하나를 고르는 것이 아니라
여러 모델을 나란히 둘 수 있게** 한 것이다. 리비전(HF 커밋 해시)도 함께 적는다 —
저장소 주인이 가중치를 갈아 끼울 수 있어 그게 곧 재현성이다.

## ⚠️ 라이선스

`snunlp/KR-FinBert-SC` 는 **라이선스 표기가 없다.** HF 모델의 약 70%가 그렇고,
관례는 라이선스 대신 **인용**이다. 이 모델은 README 에 BibTeX 을 직접 준다.
표기는 `docs/데이터파트/version3.3/모델_라이선스_대장.md` 를 따른다.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402

#: 쓰는 모델. 바꾸면 `text_signal` 에 **나란히** 쌓이지 옛 값을 덮지 않는다.
MODEL_ID = "snunlp/KR-FinBert-SC"

#: 한 번에 넣을 문장 수. CPU 라 크게 잡아도 이득이 적고 메모리만 는다.
BATCH = 64

#: 제목 최대 토큰. 실측 최대가 172글자라 64토큰이면 잘리는 일이 거의 없다.
MAX_LEN = 64

_KST = timezone(timedelta(hours=9))


def text_sha(text: str) -> str:
    """제목의 SHA-256 앞 16자. 제목 자체를 키로 쓰면 인덱스가 두껍다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def 안매긴_제목(conn: sqlite3.Connection, *, limit: int = 0) -> List[str]:
    """아직 이 모델로 안 매긴 고유 제목. 흔한 것부터 준다 — 도중에 멈춰도 많이 덮는다."""
    q = """
        SELECT d.report_nm, COUNT(*) c
          FROM dart_disclosure d
          LEFT JOIN text_signal t
            ON t.report_nm = d.report_nm AND t.model_id = ?
         WHERE d.report_nm IS NOT NULL AND d.report_nm <> '' AND t.text_sha IS NULL
         GROUP BY d.report_nm
         ORDER BY c DESC"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(q, (MODEL_ID,))]


def 현황(conn: sqlite3.Connection) -> None:
    전체 = conn.execute(
        "SELECT COUNT(DISTINCT report_nm) FROM dart_disclosure "
        "WHERE report_nm IS NOT NULL AND report_nm <> ''").fetchone()[0]
    매긴것 = conn.execute("SELECT COUNT(*) FROM text_signal WHERE model_id=?",
                          (MODEL_ID,)).fetchone()[0]
    행 = conn.execute("SELECT COUNT(*) FROM dart_disclosure").fetchone()[0]
    덮은행 = conn.execute("""
        SELECT COUNT(*) FROM dart_disclosure d
          JOIN text_signal t ON t.report_nm = d.report_nm AND t.model_id = ?""",
        (MODEL_ID,)).fetchone()[0]
    print(f"── text_signal ({MODEL_ID}) ──")
    print(f"  고유 제목 {매긴것:,} / {전체:,} "
          f"({매긴것 / 전체:.1%})" if 전체 else "  (공시가 없다)")
    print(f"  공시 행으로 치면 {덮은행:,} / {행:,} ({덮은행 / 행:.1%})" if 행 else "")
    if 매긴것:
        r = conn.execute(
            "SELECT AVG(p_neg), AVG(p_neu), AVG(p_pos), "
            "SUM(p_neu > 0.9) FROM text_signal WHERE model_id=?",
            (MODEL_ID,)).fetchone()
        print(f"  평균 확률  부정 {r[0]:.3f} · 중립 {r[1]:.3f} · 긍정 {r[2]:.3f}")
        print(f"  중립 0.9 초과인 제목 {r[3]:,} ({r[3] / 매긴것:.0%}) "
              "← 높으면 '제목에는 방향 신호가 없다' 는 뜻이다")


def 매긴다(제목들: Sequence[str]) -> Tuple[List[Dict], str, List[str]]:
    """확률 3칸을 낸다. `(행들, 리비전, 라벨순서)`."""
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        리비전 = model_info(MODEL_ID).sha
    except Exception:
        리비전 = None            # 오프라인이어도 매기는 것은 되어야 한다

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    라벨 = [model.config.id2label[i].lower()
            for i in range(model.config.num_labels)]
    # 🔴 라벨 순서를 모델에서 읽는다. 자리로 가정하면 모델을 바꿨을 때 조용히 뒤집힌다.
    칸 = {n: i for i, n in enumerate(라벨)}
    for 필요 in ("negative", "neutral", "positive"):
        if 필요 not in 칸:
            raise SystemExit(
                f"모델 라벨이 예상과 다르다: {라벨}\n"
                "  이 스크립트는 negative·neutral·positive 세 라벨을 전제한다.")

    now = datetime.now(_KST).isoformat(timespec="seconds")
    out: List[Dict] = []
    시작 = time.time()
    for i in range(0, len(제목들), BATCH):
        덩이 = list(제목들[i:i + BATCH])
        enc = tok(덩이, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        with torch.no_grad():
            확률 = torch.softmax(model(**enc).logits, dim=-1).tolist()
        for 제목, p in zip(덩이, 확률, strict=True):
            out.append({"text_sha": text_sha(제목), "report_nm": 제목,
                        "model_id": MODEL_ID, "revision": 리비전,
                        "p_neg": p[칸["negative"]], "p_neu": p[칸["neutral"]],
                        "p_pos": p[칸["positive"]], "scored_at": now})
        끝 = min(i + BATCH, len(제목들))
        if 끝 % (BATCH * 40) == 0 or 끝 == len(제목들):
            초 = time.time() - 시작
            print(f"  [{끝:>6,}/{len(제목들):,}] {초:>5.0f}초 · "
                  f"{끝 / 초:.1f}건/초 · 남은 시간 어림 "
                  f"{(len(제목들) - 끝) / (끝 / 초) / 60:.1f}분")
    return out, 리비전, 라벨


def 담는다(conn: sqlite3.Connection, 행들: Sequence[Dict]) -> int:
    if not 행들:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM text_signal").fetchone()[0]
    conn.executemany(
        "INSERT OR REPLACE INTO text_signal "
        "(text_sha, report_nm, model_id, revision, p_neg, p_neu, p_pos, scored_at) "
        "VALUES (:text_sha, :report_nm, :model_id, :revision, :p_neg, :p_neu, "
        ":p_pos, :scored_at)", 행들)
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM text_signal").fetchone()[0] - before


def main() -> int:
    ap = argparse.ArgumentParser(description="공시 제목 감성 → text_signal")
    ap.add_argument("--status", action="store_true", help="매기지 않고 현황만")
    ap.add_argument("--limit", type=int, default=0, help="이 개수의 제목만")
    args = ap.parse_args()

    conn = sqlite3.connect(krx_db_path())
    if args.status:
        현황(conn)
        return 0

    제목들 = 안매긴_제목(conn, limit=args.limit)
    print(f"── 안 매긴 고유 제목 {len(제목들):,} ──")
    if not 제목들:
        print("  매길 것이 없다.\n")
        현황(conn)
        return 0
    print(f"  모델 {MODEL_ID} · 로컬 추론 (비용 0원)\n")

    행들, 리비전, 라벨 = 매긴다(제목들)
    print(f"\n  라벨 순서 {라벨} · 리비전 {리비전}")
    담은것 = 담는다(conn, 행들)
    print(f"  담은 행 {담은것:,}\n")
    현황(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
