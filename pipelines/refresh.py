"""버튼 한 번으로 도는 갱신 — 수집부터 HF 반출까지 (이슈 #89 갈래 ②).

    python -m pipelines.refresh                # 수집 → 게이트 → 판정 → (필요할 때만) 반출·업로드
    python -m pipelines.refresh --with-adj     # 수정주가까지 다시 만든다 (약 13분)
    python -m pipelines.refresh --dry-run      # 무엇을 할지만 보고 아무것도 바꾸지 않는다
    python -m pipelines.refresh --status       # 최근 실행 기록만
    python -m pipelines.refresh --json         # 화면이 읽을 계약을 표준출력 마지막 줄에

## 왜 한 명령인가

2026-09-05 에 손으로 이 순서를 쳤다 — 수정주가 재빌드 → 종목기본정보 검사 → 반출 →
검사기 넷 → 업로드 → 카드. 여섯 번을 순서대로 쳐야 하고, 하나를 빠뜨려도 **아무 일도
일어나지 않는다.** 그날 실제로 겪은 것이 그 종류의 사고였다. 고친 코드를 좁은 범위에만
먹여서, 저장소에는 고친 코드가 있는데 나간 자료는 안 고쳐진 채였다.

한 명령으로 묶으면 빠뜨릴 수가 없고, 무엇이 어디서 멈췄는지가 한 자리에 남는다.

## 왜 수정주가는 기본으로 끄나 — 매번 돌리면 자료가 나빠진다

여기가 이 파이프라인에서 가장 헷갈리는 자리다. "최신으로 맞춘다" 는 말이 수정주가에는
반대로 작동한다.

FinanceDataReader 는 종목마다 **최근 3,000거래일**만 수정주가를 준다. 그 창은 날마다
앞으로 밀리고, 밀려난 행은 FDR 이 준 값에서 우리가 앵커에서 이어 붙인 계산값(`chain`)으로
넘어간다. 실측(2026-09-05, 창이 20140617 에 걸려 있는 상태):

    창이  20거래일 밀리면    11,502행이 fdr → chain     (반출본의 0.146%)
    창이  60거래일 밀리면    38,339행                   (0.486%)
    창이 250거래일 밀리면   158,996행                   (2.02%)

`chain` 은 우리 계산이라 오차 상한이 0.39% 로 실측돼 있다. 즉 **수정주가를 다시 돌릴
때마다 그 행들은 원본에서 근삿값으로 내려앉는다.** 게다가 값이 바뀌면 판정기가 "재배포
필요" 를 내고 404MB 를 다시 올리게 된다. 날마다 돌리면 날마다 그런다.

그리고 새로 들어오는 시세는 **전부 홀드아웃 구간(20240901~)** 이라 반출본(~20240830)을
건드리지 않는다. 그러니 날마다 수정주가를 다시 만들 이유가 애초에 없다.

수정주가를 다시 만들어야 하는 때는 **조정 코드가 바뀐 날**이다 (2026-09-05 의 감자 관문
③처럼). 그건 사람이 아는 사건이므로 `--with-adj` 로 켠다. 끈 실행도 `skipped` 로 남으므로
"안 돌렸다" 와 "돌았는데 실패했다" 는 구별된다.

## 왜 업로드가 조건부인가

`verify_hf_dataset.py` 가 HF 배포본과 지금 DB·코드를 대조해서 **재배포가 필요한지**를
판정한다(종료코드 0 = 최신 · 2 = 필요). 사람이 "바뀐 것 같으니 올리자" 로 정하지 않는다 —
새 시세만으로는 반출본이 안 바뀌는데도 매번 404MB 를 올리게 되기 때문이다.

## 왜 잠금이 필요한가

화면의 버튼은 두 번 눌린다. 두 실행이 같은 DB 에 동시에 쓰면 SQLite 가 `database is
locked` 로 **가끔만** 터지고, 그건 재현이 안 되는 버그가 된다. `filelock` 으로 막고,
두 번째 호출은 기다리지 않고 곧바로 `busy` 와 **진행 중인 run_id** 를 돌려준다 —
화면이 그 열쇠로 진행 상황을 이어서 폴링할 수 있어야 하기 때문이다.

## 진행 상황은 어디에 남나

`ingest_run` · `ingest_run_stage` 에 **시작할 때부터** 남긴다 (`pipelines/ingest.py` 와
같은 표). 화면이 폴링하는 표를 늘리지 않는다. 어느 파이프라인이 남긴 행인지는
`ingest_run.args` 의 `pipeline` 값으로 가른다.

⚠️ 표의 `status` 는 다섯 가지(`running`·`ok`·`partial`·`error`·`dry_run`)로 제한돼 있다.
   화면에 주는 계약의 `gate_failed`·`busy` 는 그 다섯에 없으므로 **표에는 `error` 로
   적고 `note` 에 까닭을 남긴다.** 표의 CHECK 를 넓히려면 마이그레이션이 필요한데,
   그 값은 표가 아니라 이 파이프라인의 사정이라 표를 고칠 일이 아니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from common.trading_calendar import today_kst  # noqa: E402
from ingest.store import run_log  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402

#: 단계 이름 — `--only` 에 쓰는 값이자 `ingest_run_stage.stage` 에 담기는 값이다.
#:
#: 순서에 뜻이 있다. `adj` 가 `ingest` **뒤**인 것은 수정주가가 새로 들어온 시세까지
#: 덮어야 하기 때문이다. 먼저 돌리면 그날 받은 날짜가 조정되지 않은 채 남는다.
STAGES = ("ingest", "adj", "gate", "verify", "export", "upload")

단계이름 = {
    "ingest": "수집",
    "adj": "수정주가",
    "gate": "품질 게이트",
    "verify": "재배포 판정",
    "export": "반출",
    "upload": "HF 업로드",
}

#: 잠금 파일. `data/` 아래에 두는 이유는 그 폴더가 이미 gitignore 되고, 저장소를
#: 받은 사람마다 자기 것이 생기면 되기 때문이다.
LOCK_PATH = Path("data/refresh.lock")

#: `ingest_run.args` 에 심어 두는 표식. 같은 표를 `pipelines/ingest.py` 와 나눠 쓰므로
#: 어느 쪽이 남긴 행인지 가를 수단이 필요하다.
PIPELINE_NAME = "refresh"

#: 수집 창의 최소값. DB 가 어제까지 차 있어도 마지막 며칠은 다시 확인한다 — 장중에
#: 받았다면 그날 자료가 확정 전일 수 있다. 이미 받은 날은 건너뛰므로 공짜다.
MIN_WINDOW = 3

#: 수집 창의 상한. 이보다 크게 필요하면 그건 "날마다 누르는 갱신" 이 아니라 대량 수집이고,
#: 버튼이 할 일이 아니다. 잘라 내지 않고 **무엇을 하라고 알려 준 뒤 세운다.**
MAX_WINDOW = 250


# ==================================================
# 창 계산 — DB 마지막 거래일에서 오늘까지
# ==================================================
def db_마지막_거래일(db: Optional[Path] = None) -> Optional[str]:
    """`daily_price` 에 담긴 마지막 날짜. 비어 있으면 `None`."""
    path = db or krx_db_path()
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT MAX(bas_dd) FROM daily_price").fetchone()
    except sqlite3.OperationalError:
        return None                      # 표가 아직 없다 — 창을 못 세니 부르는 쪽이 정한다
    finally:
        conn.close()
    return str(row[0]) if row and row[0] else None


def 창을_센다(마지막: Optional[str], 오늘: Optional[date] = None) -> Tuple[int, str]:
    """`마지막` 부터 오늘까지 며칠치를 훑어야 하나. `(거래일수, 설명)`.

    🔴 **고정 10일로 두지 않는 이유.** 열흘 넘게 안 누르면 그 앞이 조용히 빈다.
       빈 것은 다음에 누를 때도 안 채워진다 — 창이 언제나 "오늘부터 10일" 이라서다.
       그래서 창을 **DB 가 어디까지 차 있는지**에서 거꾸로 잰다.

    ⚠️ 주말만 건너뛰고 공휴일은 세지 않는다. `common.trading_calendar.trading_days` 가
       쓰는 방식과 **같아야** 하기 때문이다 — 여기서 공휴일을 빼면 그 함수가 만들어 주는
       날짜 목록보다 짧게 잡혀 실제로 받아야 할 날이 창 밖으로 밀린다. 며칠 넉넉한 것은
       공짜지만(이미 받은 날은 건너뛴다) 모자란 것은 구멍이 된다.
    """
    끝 = 오늘 or today_kst()
    if not 마지막:
        return MIN_WINDOW, "DB 가 비어 있다 — 최소 창만 본다"

    시작 = date(int(마지막[:4]), int(마지막[4:6]), int(마지막[6:]))
    if 시작 > 끝:
        # 미래 날짜가 DB 에 있다. 지어내 답하지 않고 최소 창으로 둔 뒤 그 사실을 남긴다.
        return MIN_WINDOW, f"DB 마지막({마지막})이 오늘({끝:%Y%m%d})보다 뒤다"

    평일 = 0
    d = 시작
    while d <= 끝:
        if d.weekday() < 5:
            평일 += 1
        d += timedelta(days=1)

    창 = max(MIN_WINDOW, 평일)
    설명 = f"DB {마지막} → 오늘 {끝:%Y%m%d} · 평일 {평일}일"
    return 창, 설명


# ==================================================
# 바깥 명령을 부르는 자리
# ==================================================
def _자식_환경() -> Dict[str, str]:
    """자식이 한글을 UTF-8 로, 그리고 **모아 두지 말고 곧바로** 뱉게 한다.

    `PYTHONIOENCODING` — Windows 에서 파이썬의 표준출력을 파이프로 받으면 기본 인코딩이
    cp949 가 된다. 우리 스크립트는 전부 한글로 말하므로 이걸 안 맞추면 요약 줄이 깨진 채로
    기록에 남는다. 깨진 기록은 "무엇이 실패했나" 에 답하지 못한다.

    `PYTHONUNBUFFERED` — 🔴 이게 없으면 **진행이 안 보인다.** 자식의 표준출력이 파이프라서
    파이썬이 8KB 씩 모아 쓴다. 말수가 적은 단계는 그 8KB 를 채우는 데 몇 분이 걸리고,
    그동안 우리는 한 줄도 못 받는다. 수정주가 재빌드는 13분짜리라 그 침묵이 통째로
    "멈춘 것" 처럼 보인다. 실측 2026-09-05: 이걸 넣기 전 첫 실행이 정확히 그랬다.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def 돌린다(argv: List[str], *, 꼬리: int = 3,
          timeout: int = 7200) -> Tuple[int, str]:
    """스크립트를 부르고 `(종료코드, 마지막 줄들)` 을 준다. 출력은 그대로 흘려보낸다.

    🔴 **출력을 삼키지 않는 이유.** 수정주가 재빌드는 13분 걸린다. 그동안 아무것도 안
       보이면 사람은 멈춘 것으로 읽고 창을 닫는다. 그래서 자식이 뱉는 줄을 그대로
       내보내면서, 동시에 마지막 몇 줄을 모아 기록에 남긴다.
    """
    proc = subprocess.Popen(
        [sys.executable, *argv],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=_자식_환경(),
        bufsize=1,
    )
    모은줄: List[str] = []
    assert proc.stdout is not None
    for 줄 in proc.stdout:
        줄 = 줄.rstrip()
        print(f"     │ {줄}")
        if 줄.strip():
            모은줄.append(줄.strip())
    proc.wait(timeout=timeout)
    return proc.returncode, " / ".join(모은줄[-꼬리:])


def _hf_토큰_있나() -> bool:
    """HF 쓰기 토큰이 이 PC 에 있나.

    팀장 PC 에만 있다. 팀원이 버튼을 눌러도 수집·게이트까지는 돌아야 하므로, 없으면
    판정·반출·업로드를 **건너뛴 것으로 남기고 정상 종료**한다. 실패로 적으면 팀원 화면이
    늘 붉어서 진짜 실패를 못 알아보게 된다.
    """
    try:
        from ingest.clients import hf_data
        return bool(hf_data.load_hf_key()[0])
    except Exception:                                     # noqa: BLE001
        return False


# ==================================================
# 단계마다 무엇을 할지
# ==================================================
def _단계_수집(ctx: Dict) -> Dict:
    창 = ctx["창"]
    argv = ["-m", "pipelines.ingest", "--days", str(창)]
    if ctx["dry_run"]:
        argv.append("--dry-run")
    코드, 꼬리 = 돌린다(argv)
    if 코드 != 0:
        raise RuntimeError(f"수집이 실패했다 (종료코드 {코드}) — {꼬리}")
    return {"note": f"창 {창}거래일 · {ctx['창설명']}"}


def _단계_수정주가(ctx: Dict) -> Dict:
    if not ctx["with_adj"]:
        return {"skip": True,
                "note": "조정 코드가 안 바뀌었다 (--with-adj 로 켠다). "
                        "매번 돌리면 FDR 창이 밀린 만큼 자료가 chain 으로 내려앉는다"}
    if ctx["dry_run"]:
        return {"note": "돌리면 전 종목 · 실측 13분 (2026-09-05 · 3,677종)"}
    코드, 꼬리 = 돌린다(["scripts/build_adj_prices.py"])
    if 코드 != 0:
        raise RuntimeError(f"수정주가 적재가 검증에서 멈췄다 (종료코드 {코드}) — {꼬리}")
    return {"note": 꼬리[:300]}


def _단계_게이트(ctx: Dict) -> Dict:
    if ctx["dry_run"]:
        return {"note": "돌리면 scripts/check_data.py (시세·지수 품질)"}
    코드, 꼬리 = 돌린다(["scripts/check_data.py"])
    if 코드 != 0:
        # 🔴 여기서 멈춘다. 그리고 **DB 를 되돌리지 않는다** — 수집은 원자료 보존이라
        #    되돌릴 이유가 없다. 막아야 하는 것은 자료가 아니라 **반출**이다.
        raise GateFailed(꼬리 or "품질 게이트 실패")
    return {"note": 꼬리[:300]}


def _단계_판정(ctx: Dict) -> Dict:
    if not ctx["hf"]:
        return {"skip": True, "note": "HF 토큰이 없다 — 배포본을 받아 볼 수 없다"}
    if ctx["dry_run"]:
        return {"note": "돌리면 scripts/verify_hf_dataset.py (0 최신 · 2 재배포 필요)"}
    if ctx["force_export"]:
        ctx["재배포필요"] = True
        return {"note": "--force-export — 판정을 건너뛰고 반출한다"}

    argv = ["scripts/verify_hf_dataset.py"]
    if ctx["reuse_snapshot"]:
        argv.append("--skip-download")
    코드, 꼬리 = 돌린다(argv, 꼬리=4)
    if 코드 == 0:
        ctx["재배포필요"] = False
        return {"note": "재배포가 필요 없다 — 배포본이 지금 DB·코드와 같다"}
    if 코드 == 2:
        ctx["재배포필요"] = True
        return {"note": f"재배포가 필요하다 — {꼬리}"[:300]}
    raise RuntimeError(f"판정을 못 했다 (종료코드 {코드}) — {꼬리}")


def _단계_반출(ctx: Dict) -> Dict:
    if not ctx["hf"]:
        return {"skip": True, "note": "판정을 못 해서 반출도 하지 않는다"}
    if ctx["dry_run"]:
        return {"note": "돌리면 scripts/export_team_dataset.py (개발구간만)"}
    if not ctx.get("재배포필요"):
        return {"skip": True, "note": "판정이 '최신' 이라 반출하지 않는다"}
    코드, 꼬리 = 돌린다(["scripts/export_team_dataset.py"], 꼬리=4)
    if 코드 != 0:
        raise RuntimeError(f"반출이 실패했다 (종료코드 {코드}) — {꼬리}")
    return {"note": 꼬리[:300]}


def _단계_업로드(ctx: Dict) -> Dict:
    if not ctx["hf"]:
        return {"skip": True, "note": "HF 쓰기 토큰이 없다 (팀장 PC 에만 있다)"}
    if ctx["dry_run"]:
        return {"note": "돌리면 scripts/upload_to_hf.py (검사기 4종을 스스로 돌린다)"}
    if not ctx.get("재배포필요"):
        return {"skip": True, "note": "판정이 '최신' 이라 올리지 않는다"}
    코드, 꼬리 = 돌린다(["scripts/upload_to_hf.py",
                     "--note", f"버튼 갱신 {ctx['run_id']}"], 꼬리=4)
    if 코드 != 0:
        raise RuntimeError(f"업로드가 막혔다 (종료코드 {코드}) — {꼬리}")
    ctx["올림"] = True
    return {"note": 꼬리[:300]}


단계함수 = {
    "ingest": _단계_수집,
    "adj": _단계_수정주가,
    "gate": _단계_게이트,
    "verify": _단계_판정,
    "export": _단계_반출,
    "upload": _단계_업로드,
}


class GateFailed(RuntimeError):
    """품질 게이트가 막았다. 실패와 갈라 놓는 이유는 **사람이 볼 일**이기 때문이다.

    수집 실패는 다시 누르면 될 수 있지만, 게이트 실패는 자료가 구조적으로 못 쓸 상태라는
    뜻이라 다시 눌러도 같은 곳에서 막힌다. 화면이 둘을 같은 붉은색으로 보여 주면
    사람은 일단 다시 누른다.
    """


# ==================================================
# 잠금
# ==================================================
def 진행중인_실행() -> Optional[Dict]:
    """지금 돌고 있는 refresh 실행. 없으면 `None`.

    두 번째 버튼이 **첫 번째의 열쇠**를 받아 가야 하므로 필요하다. 열쇠 없이 `busy` 만
    돌려주면 화면은 무엇을 폴링해야 할지 모른다.
    """
    path = krx_db_path()
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT run_id, started_at FROM ingest_run "
            "WHERE status = 'running' AND args LIKE ? "
            "ORDER BY started_at DESC LIMIT 1",
            (f'%"pipeline": "{PIPELINE_NAME}"%',),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return dict(row) if row else None


# ==================================================
# 실행
# ==================================================
def run(args) -> Tuple[int, Dict]:
    """`(종료코드, 화면에 줄 계약)`."""
    from filelock import FileLock, Timeout

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    잠금 = FileLock(str(LOCK_PATH), timeout=0)
    try:
        잠금.acquire()
    except Timeout:
        진행중 = 진행중인_실행()
        열쇠 = 진행중["run_id"] if 진행중 else ""
        print("이미 돌고 있다. 두 번 눌러도 한 번만 돈다.")
        if 열쇠:
            print(f"   진행 중인 실행: {열쇠} (시작 {진행중['started_at']})")
        else:
            print("   잠금은 잡혀 있는데 실행 기록이 없다 — 방금 시작했거나 "
                  "지난 실행이 기록을 남기기 전에 죽었다.")
        return 0, {"run_id": 열쇠, "status": "busy", "stages": {}, "uploaded": False}

    try:
        return _잠금_안에서(args)
    finally:
        잠금.release()


def _잠금_안에서(args) -> Tuple[int, Dict]:
    migrate_path()          # 표가 없으면 실행 기록조차 남길 수 없다

    마지막 = db_마지막_거래일()
    창, 창설명 = 창을_센다(마지막)
    if 창 > MAX_WINDOW:
        print(f"🔴 받을 구간이 너무 넓다 — {창설명}")
        print(f"   버튼 갱신은 {MAX_WINDOW}거래일까지만 본다. 이만큼 비었다면 대량 수집이다.")
        print("   할 일: python -m pipelines.ingest --days <필요한 수> 로 먼저 채운다.")
        return 1, {"run_id": "", "status": "error", "stages": {},
                   "uploaded": False, "note": "창이 상한을 넘었다"}

    run_id = run_log.new_run_id()
    ctx: Dict = {
        "run_id": run_id,
        "창": 창,
        "창설명": 창설명,
        "dry_run": bool(args.dry_run),
        "with_adj": bool(args.with_adj),
        "force_export": bool(args.force_export),
        "reuse_snapshot": bool(args.reuse_snapshot),
        "hf": _hf_토큰_있나(),
        "올림": False,
        # 🔴 판정 단계가 아니라 **여기서** 정한다. 판정 안에서만 세우면
        #    `--only export --force-export` 처럼 판정을 빼고 부를 때 아무 일도 안 일어난다
        #    (판정이 안 도니 값이 안 서고, 반출은 "판정이 최신" 으로 읽어 건너뛴다).
        #    강제한다고 했는데 조용히 아무것도 안 하는 것이 가장 나쁜 결과다.
        "재배포필요": bool(args.force_export),
    }
    실행인자 = {
        "pipeline": PIPELINE_NAME,
        "days": 창, "with_adj": ctx["with_adj"], "dry_run": ctx["dry_run"],
        "force_export": ctx["force_export"], "only": list(args.only or STAGES),
    }
    run_log.start_run(run_id, args=실행인자)

    머리 = "무엇을 할지만 본다 (아무것도 바꾸지 않는다)" if ctx["dry_run"] else "갱신을 시작한다"
    print(f"── {머리} · run_id={run_id} ──")
    print(f"   창: {창}거래일 ({창설명})")
    print(f"   수정주가: {'다시 만든다 (약 13분)' if ctx['with_adj'] else '건드리지 않는다'}")
    print(f"   HF 토큰: {'있다' if ctx['hf'] else '없다 — 판정·반출·업로드를 건너뛴다'}")
    print()

    고를단계 = list(args.only) if args.only else list(STAGES)
    상태: Dict[str, Dict] = {}
    실패단계: List[str] = []
    게이트막힘 = False
    멈춤 = False

    for stage in STAGES:
        표시 = 단계이름[stage]
        if stage not in 고를단계:
            run_log.finish_stage(run_id, stage, "skipped", note="--only 로 제외")
            상태[stage] = {"status": "skipped", "note": "--only 로 제외"}
            print(f"  ⬜ {표시:12s} 건너뜀 (--only)")
            continue
        if 멈춤:
            까닭 = "앞 단계가 멈춰서 돌지 않았다"
            run_log.finish_stage(run_id, stage, "skipped", note=까닭)
            상태[stage] = {"status": "skipped", "note": 까닭}
            print(f"  ⬜ {표시:12s} {까닭}")
            continue

        run_log.start_stage(run_id, stage)
        시작 = time.time()
        print(f"  ▶ {표시}")
        try:
            결과 = 단계함수[stage](ctx)
            걸린 = time.time() - 시작
            if 결과.get("skip"):
                run_log.finish_stage(run_id, stage, "skipped", note=결과.get("note", ""))
                상태[stage] = {"status": "skipped", "note": 결과.get("note", "")}
                print(f"  ⬜ {표시:12s} {결과.get('note', '')}")
                continue
            상태문자 = "dry_run" if ctx["dry_run"] else "ok"
            run_log.finish_stage(run_id, stage, 상태문자,
                                 rows=결과.get("rows", 0), note=결과.get("note", ""))
            상태[stage] = {"status": 상태문자, "note": 결과.get("note", ""),
                          "seconds": round(걸린, 1)}
            표식 = "🔍" if ctx["dry_run"] else "✅"
            print(f"  {표식} {표시:12s} {걸린:>6.0f}초  {결과.get('note', '')}")

        except GateFailed as exc:
            걸린 = time.time() - 시작
            까닭 = str(exc)[:400]
            run_log.finish_stage(run_id, stage, "error", note=f"gate_failed: {까닭}")
            상태[stage] = {"status": "error", "note": 까닭, "seconds": round(걸린, 1)}
            실패단계.append(stage)
            게이트막힘 = True
            멈춤 = True
            print(f"  🔴 {표시:12s} 게이트가 막았다 ({걸린:.0f}초)")
            print(f"     {까닭}")

        except Exception as exc:                          # noqa: BLE001
            걸린 = time.time() - 시작
            까닭 = f"{type(exc).__name__}: {exc}"[:400]
            run_log.finish_stage(run_id, stage, "error", note=까닭)
            상태[stage] = {"status": "error", "note": 까닭, "seconds": round(걸린, 1)}
            실패단계.append(stage)
            # 🔴 여기서 멈추는 것이 `pipelines/ingest.py` 와 다른 점이다.
            #    수집은 거시가 실패해도 시세는 받아야 하지만, 여기 단계들은 **앞의 결과를
            #    딛고 선다.** 수집이 실패한 채로 반출하면 반쪽짜리가 팀에 나간다.
            멈춤 = True
            print(f"  🔴 {표시:12s} 실패 ({걸린:.0f}초)")
            print(f"     {까닭}")

    if ctx["dry_run"]:
        최종, 계약상태 = "dry_run", "dry_run"
    elif 게이트막힘:
        최종, 계약상태 = "error", "gate_failed"
    elif 실패단계:
        최종, 계약상태 = "error", "error"
    else:
        최종, 계약상태 = "ok", "ok"
    run_log.finish_run(run_id, 최종,
                       note=f"실패 단계: {' '.join(실패단계)}" if 실패단계 else "")

    print()
    print(f"── {계약상태} · run_id={run_id} ──")
    if ctx["올림"]:
        print("   HF 에 새 커밋이 올라갔다")
    elif not ctx["dry_run"] and not 실패단계:
        print("   HF 는 그대로다 (판정이 '최신' 이거나 토큰이 없다)")
    if 게이트막힘:
        print("   🔴 게이트 실패는 다시 눌러도 같은 곳에서 막힌다 — 자료를 봐야 한다.")
        print("      python scripts/check_data.py  로 무엇이 붉은지 본다")

    계약 = {"run_id": run_id, "status": 계약상태, "stages": 상태,
           "uploaded": bool(ctx["올림"])}
    return (0 if 최종 in ("ok", "dry_run") else 1), 계약


# ==================================================
# 조회
# ==================================================
def 최근_실행을_보여준다(limit: int) -> int:
    실행들 = [r for r in run_log.latest_runs(limit * 4)
            if f'"pipeline": "{PIPELINE_NAME}"' in (r.get("args") or "")]
    if not 실행들:
        print("버튼 갱신 실행 기록이 아직 없다.")
        print("  python -m pipelines.refresh --dry-run  으로 먼저 무엇을 할지 본다.")
        return 0
    for r in 실행들[:limit]:
        print(f"── {r['run_id']} · {r['status']} · {r['started_at']} ──")
        for s in r.get("stages", []):
            표시 = 단계이름.get(s["stage"], s["stage"])
            print(f"   {표시:12s} {s['status']:8s} {s.get('note') or ''}")
        print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipelines.refresh",
        description="수집 → 게이트 → 판정 → (필요할 때만) 반출·HF 업로드를 한 명령으로")
    parser.add_argument("--only", nargs="+", choices=STAGES, metavar="단계",
                        help=f"고를 단계: {' · '.join(STAGES)}")
    parser.add_argument("--with-adj", action="store_true",
                        help="수정주가를 다시 만든다 (약 13분). "
                             "🔴 조정 코드가 바뀐 날에만 켠다 — 매번 돌리면 FDR 창이 "
                             "밀린 만큼 옛 행이 원본에서 우리 계산값으로 내려앉는다")
    parser.add_argument("--force-export", action="store_true",
                        help="재배포 판정을 건너뛰고 반출·업로드한다 "
                             "(판정이 404MB 를 받는 것을 아낄 때)")
    parser.add_argument("--reuse-snapshot", action="store_true",
                        help="판정에서 이미 받아 둔 스냅샷을 그대로 쓴다 "
                             "(data/outbox/hf_snapshot 이 있어야 한다)")
    parser.add_argument("--dry-run", action="store_true",
                        help="무엇을 할지만 보고 아무것도 바꾸지 않는다")
    parser.add_argument("--status", action="store_true",
                        help="돌리지 않고 최근 실행 기록만 보여준다")
    parser.add_argument("--limit-runs", type=int, default=3,
                        help="--status 에서 보여줄 실행 수 (기본 3)")
    parser.add_argument("--json", action="store_true",
                        help="화면이 읽을 계약을 표준출력 마지막 줄에 한 줄로 낸다")
    args = parser.parse_args(argv)

    # 🔴 줄 단위로 흘려보낸다. 안 그러면 **진행이 안 보인다.**
    #
    # 파이썬은 표준출력이 터미널이 아닐 때(파일·파이프로 받을 때) 8KB 씩 모아서 쓴다.
    # 화면은 이 명령을 `subprocess.Popen` 으로 부르므로 정확히 그 경우에 해당하고,
    # 수정주가를 켠 실행은 13분 내내 **한 줄도 안 나온다.** 그러면 사람은 멈춘 것으로
    # 읽고 버튼을 다시 누르거나 창을 닫는다. 2026-09-05 첫 실행에서 실제로 그랬다.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass                          # 되감을 수 없는 스트림이면 그냥 둔다

    if args.status:
        return 최근_실행을_보여준다(args.limit_runs)

    코드, 계약 = run(args)
    if args.json:
        print(json.dumps(계약, ensure_ascii=False))
    return 코드


if __name__ == "__main__":
    raise SystemExit(main())
