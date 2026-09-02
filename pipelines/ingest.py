"""매일 돌리는 수집 한 명령 — 시세·지수·재무·거시.

    python -m pipelines.ingest                    # 전부
    python -m pipelines.ingest --only macro       # 골라서
    python -m pipelines.ingest --only price macro
    python -m pipelines.ingest --dry-run          # 무엇을 할지만 보고 받지는 않는다
    python -m pipelines.ingest --status           # 최근 실행 기록만

## 왜 스크립트를 따로 치지 않고 한 명령인가

지금은 매일 `fetch_krx.py` · `fetch_index.py` · `fetch_dart.py` · `fetch_macro.py` 를
차례로 쳐야 한다. 하나를 빠뜨려도 **아무 일도 일어나지 않는다** — 그 자료만 어제
상태로 남고, 며칠 뒤 "왜 거시만 오래됐지" 를 되짚게 된다.

한 명령으로 묶으면 빠뜨릴 수가 없고, **무엇이 실패했는지가 한 자리에 남는다.**

## 왜 `subprocess` 로 스크립트를 부르지 않는가

부를 수도 있지만 그러면 진행 상황이 표준출력 문자열로만 오고, 그걸 다시 파싱해야
한다. 파싱은 언젠가 실패한다. 저장 계층(`krx_store` · `dart_store` · `macro_store`)을
직접 부르면 결과를 **자료구조 그대로** 받아 `ingest_run_stage` 에 담을 수 있다.

(`api/` 의 FastAPI 는 반대로 이 CLI 를 `subprocess` 로 부른다. 그쪽은 오래 도는 작업을
요청 스레드에서 떼어 놓는 것이 목적이라 경계가 프로세스여야 한다.)

## 한 단계가 실패해도 나머지는 돈다

거시가 실패했다고 시세까지 멈추면 하루치가 통째로 빈다. 단계마다 잡아서 `error` 로
남기고 다음으로 간다. 실행 전체는 `partial` 로 끝나고, 종료 코드는 1 이다 —
작업 스케줄러가 실패를 알아챌 수 있어야 한다.

## 진행 상황은 DB 에 남는다

`ingest_run` · `ingest_run_stage` 에 **시작할 때부터** 남긴다. 끝나고 한꺼번에 쓰면
중간에 죽었을 때 아무것도 안 남아서 "애초에 안 돌았다" 와 구별되지 않는다.
팀원 대시보드는 이 두 표를 폴링한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.trading_calendar import today_kst, trading_days  # noqa: E402
from ingest.store import (  # noqa: E402
    dart_store,
    krx_index,
    krx_store,
    macro_store,
    run_log,
)
from ingest.store.migrations import migrate_path  # noqa: E402

#: 단계 이름 — `--only` 에 쓰는 값이자 `ingest_run_stage.stage` 에 담기는 값이다.
STAGES = ("price", "index", "financial", "macro")

#: 기본으로 훑는 거래일 수. 매일 도는 것을 전제로 넉넉히 잡는다 —
#: 연휴로 며칠 못 돌아도 빈 날이 생기지 않는다. 이미 받은 날은 건너뛰므로 공짜다.
DEFAULT_DAYS = 10


# ==================================================
# 단계마다 무엇을 할지
# ==================================================
def _받을_날짜(days: int, end: Optional[str], 이미받음: set) -> List[str]:
    """아직 안 받은 거래일. `krx_store.sync` 가 안에서 하는 계산과 **같은 방식**이다.

    dry-run 이 "몇 개를 받을 것인가" 에 답하려면 세어 봐야 하는데, sync 는 세기만 하는
    길을 열어 두지 않았다. 복제한 셈이라 한쪽만 바뀌면 어긋난다 — 그래서 여기서 나온
    수는 **어림수로만 쓰고**, 실제로 무엇을 받을지는 언제나 sync 가 정한다.
    """
    기준 = datetime.strptime(end, "%Y%m%d").date() if end else today_kst()
    바라는날 = [d.strftime("%Y%m%d") for d in trading_days(days, end=기준)]
    return [d for d in 바라는날 if d not in 이미받음]


def _단계_시세(args, dry_run: bool) -> Dict:
    if dry_run:
        받을날 = _받을_날짜(args.days, args.end, krx_store.fetched_dates())
        return {"rows": 0,
                "note": f"최근 {args.days}거래일 중 {len(받을날)}일이 미수집 "
                        f"(시장 {len(krx_store.MARKETS)}곳 × {len(받을날)}콜)"}
    결과 = krx_store.sync(days=args.days, workers=args.workers, end=args.end)
    note = (f"요청 {결과['requested']}일 · 기존 {결과['already']}일 · "
            f"신규 {결과['fetched']}일")
    if 결과.get("failed"):
        note += f" · 실패 {len(결과['failed'])}일"
    return {"rows": 결과["rows"], "note": note, "failed": 결과.get("failed") or []}


def _단계_지수(args, dry_run: bool) -> Dict:
    if dry_run:
        # 지수는 시장마다 받은 날짜가 다르다. 가장 덜 받은 시장을 기준으로 센다 —
        # 적게 잡으면 "받을 게 없다" 고 잘못 안심하게 된다.
        시장들 = list(krx_index.DEFAULT_MARKETS)
        받을수 = max(
            len(_받을_날짜(args.days, args.end, krx_index.fetched_dates(m)))
            for m in 시장들
        )
        return {"rows": 0,
                "note": f"최근 {args.days}거래일 중 최대 {받을수}일이 미수집 "
                        f"(시장 {len(시장들)}곳)"}
    결과 = krx_index.sync(days=args.days, end=args.end)
    note = (f"요청 {결과['requested']}일 · 기존 {결과['already']}일 · "
            f"신규 {결과['fetched']}일")
    if 결과.get("failed"):
        note += f" · 실패 {len(결과['failed'])}일"
    return {"rows": 결과["rows"], "note": note, "failed": 결과.get("failed") or []}


def _단계_재무(args, dry_run: bool) -> Dict:
    연도들 = [int(y) for y in str(args.years).split(",") if y.strip()]
    if dry_run:
        return {"rows": 0,
                "note": f"유니버스 × {len(연도들)}개년({args.years}) 중 미수집분 · "
                        f"회사·연도마다 2콜"}
    결과 = dart_store.sync(years=연도들, limit=args.limit)
    note = (f"요청 {결과['requested']} · 기존 {결과['already']} · ok {결과['ok']} · "
            f"empty {결과['empty']} · error {결과['error']}")
    if 결과.get("stopped"):
        note += f" · 멈춤({결과['stopped']})"
    return {"rows": 결과["rows"], "note": note}


def _단계_거시(args, dry_run: bool) -> Dict:
    if dry_run:
        이름들 = list(macro_store.all_indicators())
        return {"rows": 0,
                "note": f"{len(이름들)}종을 다시 받는다(개정 반영) · 지표당 1콜"}
    결과 = macro_store.sync(years=args.macro_years)
    note = (f"ok {결과['ok']} · empty {결과['empty']} · error {결과['error']}")
    if 결과["error"]:
        실패 = [d["id"] for d in 결과["details"] if d["status"] == "error"]
        note += f" · 실패: {' '.join(실패)}"
    return {"rows": 결과["rows"], "note": note,
            "failed": [d for d in 결과["details"] if d["status"] == "error"]}


단계함수: Dict[str, Callable] = {
    "price": _단계_시세,
    "index": _단계_지수,
    "financial": _단계_재무,
    "macro": _단계_거시,
}

단계이름 = {
    "price": "시세 (KRX 일별)",
    "index": "지수 (KRX 지수)",
    "financial": "재무 (DART)",
    "macro": "거시 (ECOS)",
}


# ==================================================
# 실행
# ==================================================
def run(args) -> int:
    고를단계: List[str] = list(args.only) if args.only else list(STAGES)

    # 스키마를 먼저 맞춘다. 표가 없으면 실행 기록조차 남길 수 없다.
    migrate_path()

    run_id = run_log.new_run_id()
    실행인자 = {
        "only": 고를단계, "days": args.days, "years": args.years,
        "macro_years": args.macro_years, "dry_run": bool(args.dry_run),
    }
    run_log.start_run(run_id, args=실행인자)

    머리 = "무엇을 할지만 본다 (실제로 받지 않는다)" if args.dry_run else "수집을 시작한다"
    print(f"── {머리} · run_id={run_id} ──")
    print(f"   단계: {' → '.join(고를단계)}")
    print()

    실패단계: List[str] = []
    전체행 = 0

    for stage in STAGES:
        표시 = 단계이름[stage]
        if stage not in 고를단계:
            # 건너뛴 것도 남긴다. 안 남기면 "안 돌았다" 와 "실패했다" 가 같아 보인다.
            run_log.finish_stage(run_id, stage, "skipped", note="--only 로 제외")
            print(f"  ⬜ {표시:20s} 건너뜀")
            continue

        run_log.start_stage(run_id, stage)
        시작 = time.time()
        try:
            결과 = 단계함수[stage](args, args.dry_run)
            걸린 = time.time() - 시작
            상태 = "dry_run" if args.dry_run else "ok"
            run_log.finish_stage(run_id, stage, 상태,
                                 rows=결과.get("rows", 0), note=결과.get("note", ""))
            전체행 += 결과.get("rows", 0)
            표식 = "🔍" if args.dry_run else "✅"
            print(f"  {표식} {표시:20s} {결과.get('rows', 0):>9,}행  "
                  f"{걸린:>5.0f}초  {결과.get('note', '')}")

            # 단계 안에서 일부가 실패한 경우 — 단계 자체는 끝났지만 알려야 한다.
            if 결과.get("failed"):
                print(f"     ↳ 일부 실패 {len(결과['failed'])}건")

        except Exception as exc:                          # noqa: BLE001 — 한 단계의 실패가
            걸린 = time.time() - 시작                      # 나머지를 멈추면 안 된다
            run_log.finish_stage(run_id, stage, "error",
                                 note=f"{type(exc).__name__}: {exc}"[:400])
            실패단계.append(stage)
            print(f"  🔴 {표시:20s} 실패 ({걸린:.0f}초)")
            print(f"     {type(exc).__name__}: {str(exc)[:200]}")

    # ── 마무리 ──
    if args.dry_run:
        최종 = "dry_run"
    elif 실패단계:
        최종 = "partial" if len(실패단계) < len(고를단계) else "error"
    else:
        최종 = "ok"
    run_log.finish_run(run_id, 최종,
                       note=f"실패 단계: {' '.join(실패단계)}" if 실패단계 else "")

    print()
    print(f"── {최종} · {전체행:,}행 · run_id={run_id} ──")
    if 실패단계:
        print(f"   실패한 단계: {' · '.join(실패단계)}")
        print("   무엇이 실패했는지: python -m pipelines.ingest --status")
    return 1 if 실패단계 else 0


def 최근_실행을_보여준다(limit: int) -> int:
    실행들 = run_log.latest_runs(limit=limit)
    if not 실행들:
        print("아직 실행 기록이 없다.")
        print("  할 일: python -m pipelines.ingest")
        return 0

    for run_ in 실행들:
        끝 = run_["finished_at"] or "(안 끝남)"
        print(f"── {run_['run_id']} · {run_['status']} ──")
        print(f"   {run_['started_at']} → {끝}")
        if run_["note"]:
            print(f"   {run_['note']}")
        for s in run_["stages"]:
            표식 = {"ok": "✅", "error": "🔴", "skipped": "⬜",
                    "running": "⏳", "dry_run": "🔍"}.get(s["status"], "  ")
            print(f"   {표식} {s['stage']:10s} {s['rows']:>9,}행  {s['note'] or ''}")
        print()

    죽은것 = run_log.stale_runs()
    if 죽은것:
        print(f"🔴 돌다 죽은 것으로 보이는 실행 {len(죽은것)}건 "
              f"(6시간 넘게 running):")
        for run_ in 죽은것:
            print(f"   {run_['run_id']} · {run_['started_at']} 시작")
        print("   기록을 지우지 않는 이유: 지우면 '안 돌았다' 와 구별되지 않는다.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipelines.ingest",
        description="시세·지수·재무·거시를 한 명령으로 받는다")
    parser.add_argument("--only", nargs="+", choices=STAGES, metavar="단계",
                        help=f"고를 단계: {' · '.join(STAGES)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="무엇을 할지만 보고 실제로 받지 않는다")
    parser.add_argument("--status", action="store_true",
                        help="받지 않고 최근 실행 기록만 보여준다")
    parser.add_argument("--limit-runs", type=int, default=3,
                        help="--status 에서 보여줄 실행 수 (기본 3)")

    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"시세·지수에서 훑을 거래일 수 (기본 {DEFAULT_DAYS})")
    parser.add_argument("--workers", type=int, default=6,
                        help="시세 동시 호출 수 (기본 6)")
    parser.add_argument("--end", default=None,
                        help="기준일 YYYYMMDD (기본: 오늘)")
    parser.add_argument("--years", default="2021,2022,2023,2024,2025",
                        help="재무 사업연도 (쉼표로 구분)")
    parser.add_argument("--limit", type=int, default=None,
                        help="재무에서 받을 회사 수 상한 (시험용)")
    parser.add_argument("--macro-years", type=int, default=macro_store.DEFAULT_YEARS,
                        help=f"거시에서 거슬러 올라갈 햇수 "
                             f"(기본 {macro_store.DEFAULT_YEARS})")

    args = parser.parse_args(argv)

    if args.status:
        return 최근_실행을_보여준다(args.limit_runs)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
