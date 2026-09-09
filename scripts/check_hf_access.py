"""HF 에서 자료를 못 받을 때 **어디서 막혔는지** 짚어 준다.

왜 필요한가
----------
막히는 자리가 여섯 군데인데 **에러 메시지는 거의 다 `404`** 로 똑같이 보인다.
특히 이 둘이 구분되지 않는다.

    repo_type="dataset" 을 안 줬다   → RepositoryNotFoundError: 404
    파일 경로에 small/ 을 안 붙였다  → EntryNotFoundError: 404

앞의 것은 "우리 저장소가 없다" 로 읽히지만 실은 **모델 저장소를 찾은 것**이다
(`hf_hub_download` 의 기본 `repo_type` 이 `model` 이다). 사람이 이 차이를 눈으로
가리기는 어렵다. 그래서 한 단계씩 짚어 **무엇을 해야 하는지까지** 출력한다.

멈출 때 이유만 말하고 끝내지 않는다 — 다음에 할 일을 함께 적는다.

쓰는 법
------
    python scripts/check_hf_access.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = "qurious-quant/alphastack-krx-dev"
ORG = "qurious-quant"
#: 작아서 받아 보기 좋은 파일 (1KB)
PROBE = "small/sample_codes.json"


def 짚기(단계: str, 결과: bool, 안내: str = "") -> bool:
    print(f"  {'✅' if 결과 else '🔴'} {단계}")
    if not 결과 and 안내:
        for line in 안내.strip("\n").splitlines():
            print(f"       {line}")
    return 결과


def main() -> int:
    print(f"HF 접근 진단 — {REPO}\n")

    # ── 1. 라이브러리 ────────────────────────────────────────────────
    try:
        import huggingface_hub
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        짚기("huggingface_hub 설치", False, """
            uv pip install huggingface_hub --python .venv/Scripts/python.exe
              (uv 를 안 쓰시면)  pip install huggingface_hub
        """)
        return 1
    짚기(f"huggingface_hub 설치 (v{huggingface_hub.__version__})", True)

    # ── 2. 토큰 ──────────────────────────────────────────────────────
    from common import secrets
    token, source = secrets.load_key(
        ["HUGGINGFACE_ACCESS_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"])
    if not token:
        token = os.environ.get("HF_TOKEN", "")
        source = "환경변수" if token else "none"
    if not 짚기(f"토큰 (출처: {source})", bool(token), f"""
        1) https://huggingface.co/settings/tokens 에서 New token → **Read** 권한
        2) 프로젝트 루트의 .env 에 한 줄 추가:
               HUGGINGFACE_ACCESS_TOKEN=hf_본인토큰
        3) .env 는 gitignore 됩니다. 남의 토큰을 받아 쓰지 마세요.
        지금 찾아본 곳: .env · .key · 환경변수(HF_TOKEN)
        현재 폴더: {Path.cwd()}
    """):
        return 1

    api = HfApi(token=token)

    # ── 3. 내가 누구인가 ─────────────────────────────────────────────
    try:
        who = api.whoami()
    except Exception as e:                          # noqa: BLE001
        짚기("토큰이 살아 있는가", False, f"""
            {type(e).__name__}: {str(e)[:120]}
            → 토큰이 만료됐거나 지워졌습니다. 새로 만들어 .env 를 고쳐 주세요.
        """)
        return 1
    이름 = who.get("name")
    짚기(f"로그인 — {이름} ({who.get('fullname', '')})", True)

    # ── 4. 조직 멤버인가 ─────────────────────────────────────────────
    orgs = [o["name"] for o in who.get("orgs", [])]
    if not 짚기(f"조직 {ORG} 멤버 (내 소속: {orgs or '없음'})", ORG in orgs, f"""
        이 저장소는 private 이라 조직 멤버만 받을 수 있습니다.
        → 이슈 #29 에 **HF 아이디 '{이름}'** 를 댓글로 남겨 주세요. 초대해 드립니다.
    """):
        return 1

    # ── 5. 저장소가 보이는가 ─────────────────────────────────────────
    try:
        info = api.repo_info(repo_id=REPO, repo_type="dataset")
    except Exception as e:                          # noqa: BLE001
        짚기("저장소 조회", False, f"""
            {type(e).__name__}: {str(e)[:120]}
            → 조직 멤버인데도 404 라면 저장소 이름을 확인해 주세요:
                 {REPO}
        """)
        return 1
    파일들 = sorted(s.rfilename for s in info.siblings)
    짚기(f"저장소 조회 — 파일 {len(파일들)}개 · private={info.private}", True)

    # ── 6. 실제로 받아지는가 ─────────────────────────────────────────
    try:
        path = hf_hub_download(repo_id=REPO, repo_type="dataset",
                               filename=PROBE, token=token)
    except Exception as e:                          # noqa: BLE001
        짚기(f"내려받기 ({PROBE})", False, f"""
            {type(e).__name__}: {str(e)[:150]}
            → 네트워크·프록시 문제일 수 있습니다. 잠시 뒤 다시 시도해 보세요.
        """)
        return 1
    짚기(f"내려받기 — {Path(path).stat().st_size:,} B", True)

    # ── 통과 ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ✅ 여기까지 왔으면 접근에는 문제가 없습니다.")
    print("=" * 62)
    print("""
  못 받으셨다면 **부르는 방법** 쪽입니다. 아래 두 가지가 가장 흔합니다.

  ① repo_type 을 빠뜨림 — 기본값이 "model" 이라 404 가 납니다
       hf_hub_download(repo_id=..., filename=...)                  🔴 404
       hf_hub_download(repo_id=..., repo_type="dataset", ...)      ✅

  ② 폴더 접두사를 빠뜨림 — full/ · small/ 이 필요합니다
       filename="daily_price_dev.parquet"                          🔴 404
       filename="full/daily_price_dev.parquet"                     ✅

  받을 수 있는 파일 목록:""")
    for f in 파일들:
        if not f.startswith("."):
            print(f"       {f}")
    print("""
  바로 되는 예:

      from huggingface_hub import hf_hub_download
      import pandas as pd

      p = hf_hub_download(
          repo_id="qurious-quant/alphastack-krx-dev",
          repo_type="dataset",                      # 🔴 빠뜨리면 404
          filename="full/daily_price_dev.parquet",  # 🔴 full/ 필요
      )
      df = pd.read_parquet(p)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
