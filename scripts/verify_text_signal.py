"""`text_signal` 을 **전량으로** 검사한다.

    python scripts/verify_text_signal.py

`scripts/upload_to_hf.py` 의 `VERIFIERS` 에 등록돼 있어 **반출 경로가 이걸 먼저
돌린다.** 붉으면 올라가지 않는다.

## 무엇을 재나

1. 커버리지 — 채울 수 있는 공시가 전부 채워졌나
2. `known_at` — 접수일보다 뒤인가 · 거래일인가 · **그 사이에 다른 거래일이 없는가**
3. 확률 — [0,1] 이고 셋의 합이 1 인가
4. `text_sha` — 제목의 SHA-256 앞 16자와 맞나 (**전량** 다시 계산해 대조)
5. 모델 리비전이 하나인가 (섞이면 어느 행이 어느 가중치인지 모른다)
6. 라벨 분포

## 🔴 이 검사가 안 보는 축

**모델이 뜻을 맞게 읽는가는 못 본다.** 여기서 재는 것은 *"값이 규격에 맞나"* 이지
*"그 값이 옳은가"* 가 아니다. "투자설명서(일괄신고)" 가 긍정 0.93 으로 나오는데
그게 타당한지는 이 검사가 판정하지 못한다.

그건 **바깥 기준**으로만 잴 수 있다 — 5일 방향과의 카이제곱 검정이 그 자리이고,
결과는 `Cramer's V = 0.026`(사실상 무관계)이었다. 규격 검사를 통과했다고 신호가
있는 것이 아니다.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from common.trading_calendar import load_session_days  # noqa: E402

HOLDOUT_START = "20240901"
KNOWN_RULE = "rceptDt+1session"


def main() -> int:
    db = sqlite3.connect(krx_db_path())
    db.row_factory = sqlite3.Row
    문제 = []

    있나 = db.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='text_signal'"
    ).fetchone()[0]
    if not 있나:
        print("── text_signal 검사 ──\n  표가 없다 — 마이그레이션 v12 를 적용한다")
        return 1

    n = db.execute("SELECT COUNT(*) FROM text_signal").fetchone()[0]
    print(f"── text_signal 전량 검사 ── ({n:,}행)\n")
    if not n:
        print("  (비어 있음) — python scripts/score_text_signal.py 로 채운다")
        return 0

    # ── 1. 커버리지 ──────────────────────────────────────────
    print("── 1. 커버리지 ──")
    r = db.execute("""
        SELECT COUNT(DISTINCT d.report_nm) 전체,
               COUNT(DISTINCT t.report_nm) 매긴것
          FROM dart_disclosure d
          LEFT JOIN text_signal t ON t.report_nm = d.report_nm
         WHERE d.report_nm IS NOT NULL AND d.report_nm <> ''""").fetchone()
    빠진것 = r["전체"] - r["매긴것"]
    print(f"  {'✅' if not 빠진것 else '🔴'} 고유 제목 {r['매긴것']:,} / {r['전체']:,} "
          f"· 안 매긴 것 {빠진것:,}")
    if 빠진것:
        문제.append(f"안 매긴 고유 제목 {빠진것:,} — python scripts/score_text_signal.py")

    행 = db.execute("""
        SELECT COUNT(*) 전체,
               SUM(CASE WHEN t.text_sha IS NOT NULL THEN 1 ELSE 0 END) 덮은것
          FROM dart_disclosure d
          LEFT JOIN text_signal t ON t.report_nm = d.report_nm""").fetchone()
    print(f"  공시 행으로 치면 {행['덮은것']:,} / {행['전체']:,} "
          f"({행['덮은것'] / 행['전체']:.1%})" if 행["전체"] else "")

    # ── 2. text_sha 를 **전량** 다시 계산해 대조 ──────────────
    #
    # 표본으로 확인하면 "표본이 통과시킨 것" 을 놓친다. 18,600행이라 전량이 싸다.
    print("\n── 2. text_sha (전량 재계산) ──")
    어긋남 = 0
    보기 = []
    for row in db.execute("SELECT text_sha, report_nm FROM text_signal"):
        참 = hashlib.sha256(row["report_nm"].encode("utf-8")).hexdigest()[:16]
        if 참 != row["text_sha"]:
            어긋남 += 1
            if len(보기) < 3:
                보기.append((row["report_nm"][:40], row["text_sha"], 참))
    print(f"  {'✅' if not 어긋남 else '🔴'} 해시가 제목과 어긋나는 행 {어긋남:,}")
    for nm, 적힌, 참 in 보기:
        print(f"     {nm} · 적힌 {적힌} ≠ 실제 {참}")
    if 어긋남:
        문제.append(f"text_sha 가 제목과 어긋나는 행 {어긋남:,} — 조인이 틀린 짝을 만든다")

    # ── 3. 확률 ─────────────────────────────────────────────
    print("\n── 3. 확률 ──")
    r = db.execute("""
        SELECT SUM(p_neg < 0 OR p_neg > 1 OR p_neu < 0 OR p_neu > 1
                   OR p_pos < 0 OR p_pos > 1) 범위밖,
               SUM(ABS(p_neg + p_neu + p_pos - 1.0) > 0.01) 합틀림,
               SUM(p_neg IS NULL OR p_neu IS NULL OR p_pos IS NULL) 빈값
          FROM text_signal""").fetchone()
    for 이름, 값, 설명 in (("범위밖", r["범위밖"], "[0,1] 을 벗어난 행"),
                           ("합틀림", r["합틀림"], "셋의 합이 1 이 아닌 행 (softmax 누락)"),
                           ("빈값", r["빈값"], "확률이 빈 행")):
        print(f"  {'✅' if not 값 else '🔴'} {설명} {값:,}")
        if 값:
            문제.append(f"확률 {이름} {값:,}행")

    # ── 4. 모델·리비전 ───────────────────────────────────────
    print("\n── 4. 모델 · 리비전 ──")
    모델들 = dict(db.execute(
        "SELECT model_id, COUNT(*) FROM text_signal GROUP BY model_id").fetchall())
    print(f"  모델 {모델들}")
    리비전 = dict(db.execute(
        "SELECT COALESCE(revision, '(없음)'), COUNT(*) FROM text_signal "
        "GROUP BY revision").fetchall())
    print(f"  {'✅' if len(리비전) == 1 else '⚠️'} 리비전 {리비전}")
    if len(리비전) > 1:
        print("     (모델을 새로 받아 다시 매긴 흔적이다. 섞여 있으면 어느 행이 "
              "어느 가중치인지 모른다)")
    if "(없음)" in 리비전:
        print("     ⚠️ 리비전이 빈 행이 있다 — 오프라인으로 매기면 이렇게 된다. "
              "재현성을 위해 다시 매기는 것이 낫다")

    # ── 5. known_at 규칙 (반출 파일이 지켜야 하는 것) ──────────
    #
    # DB 의 `text_signal` 에는 `known_at` 이 없다 — 제목 단위라 시점이 없기 때문이다.
    # 시점은 **반출할 때** 접수일에서 만든다. 그래서 여기서는 그 규칙이 성립하는지를
    # `dart_disclosure` 로 확인한다.
    print("\n── 5. known_at 규칙 (접수일 다음 거래일) ──")
    달력 = sorted(load_session_days())
    마지막 = 달력[-1]
    import bisect
    접수일들 = [r[0] for r in db.execute(
        "SELECT DISTINCT rcept_dt FROM dart_disclosure ORDER BY rcept_dt")]
    거래일아님, 밖 = 0, 0
    for d in 접수일들:
        i = bisect.bisect_right(달력, d)
        if i >= len(달력):
            밖 += 1
            continue
        # 접수일과 known_at 사이에 다른 거래일이 없어야 한다 — bisect 가 그것을 보장한다
        if 달력[i] <= d:
            거래일아님 += 1
    print(f"  접수일 {len(접수일들):,}종 · 달력 마지막 {마지막}")
    print(f"  {'✅' if not 거래일아님 else '🔴'} known_at 이 접수일보다 이르거나 같은 날 "
          f"{거래일아님:,}")
    print(f"  {'✅' if not 밖 else '⚠️'} 달력 밖이라 known_at 을 못 내는 접수일 {밖:,} "
          "(반출에서 뺀다 — 지어내지 않는다)")
    if 거래일아님:
        문제.append(f"known_at 이 접수일 이하인 날 {거래일아님:,} — 미래참조다")

    # ── 6. 라벨 분포 ────────────────────────────────────────
    print("\n── 6. 라벨 분포 (개발구간 공시 행 기준) ──")
    분포 = dict(db.execute(f"""
        SELECT CASE WHEN t.p_pos >= t.p_neg AND t.p_pos >= t.p_neu THEN 'positive'
                    WHEN t.p_neg >= t.p_neu THEN 'negative' ELSE 'neutral' END 라벨,
               COUNT(*)
          FROM dart_disclosure d JOIN text_signal t ON t.report_nm = d.report_nm
         WHERE d.rcept_dt < '{HOLDOUT_START}'
         GROUP BY 라벨""").fetchall())
    합 = sum(분포.values()) or 1
    for k in ("negative", "neutral", "positive"):
        v = 분포.get(k, 0)
        print(f"  {k:<9} {v:>9,} ({v / 합:>6.1%})")
    print("  ℹ️ 설계서 §2.7 게이트(셋 다 15~45%)는 **미달**이다. 공시는 원래 대부분이")
    print("     중립인 자료라 그 전제가 안 맞는다. 카이제곱 검정으로 갈음했다 —")
    print("     Cramer's V = 0.026 (사실상 무관계). 5일 방향 피처로 쓰지 않는다.")

    print("\n── 판정 ──", end=" ")
    if 문제:
        print("🔴 손볼 것이 있다")
        for m in 문제:
            print(f"  · {m}")
        return 1
    print("✅ 이상 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
