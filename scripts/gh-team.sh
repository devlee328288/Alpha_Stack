#!/usr/bin/env bash
# 팀 계정으로 gh 를 한 번만 실행한다 — 전역 활성 계정을 바꾸지 않는다.
#
# 왜 이게 필요한가
# ----------------
# `git` 신원은 `~/.gitconfig` 의 `includeIf` 가 폴더 단위로 갈라 주지만(AGENTS.md 3절),
# **`gh` 는 그걸 모른다.** gh 는 전역 "활성 계정" 하나만 본다. 그래서 이 폴더에서
# `gh pr create` 를 하면 개인 계정으로 나가고 `must be a collaborator` 로 실패한다(실측).
#
# `gh auth switch` 로 바꿀 수도 있지만 AGENTS.md 3절이 그걸 금지한다 —
# 전역 상태를 바꿔 두면 **다른 창에서 개인 저장소를 만질 때 팀 계정으로 올라간다.**
# 커밋·push·PR 을 여러 저장소에서 동시에 하는 상황에서 실제로 위험하다.
#
# 이 스크립트는 키링에서 팀 토큰을 그때그때 꺼내 **그 한 번의 실행에만** 쓴다.
# 토큰을 파일에 적지 않으므로 PUBLIC 저장소에 올라갈 것이 없다.
#
# 사용법
# ------
#     bash scripts/gh-team.sh pr create --fill
#     bash scripts/gh-team.sh pr list
#     bash scripts/gh-team.sh repo view
#
# ⚠️ `gh pr merge` 는 쓰지 않는다 — 계정 정지 이력이 있다(AGENTS.md 1.5).
#    머지는 사람이 GitHub 웹에서 한다.

set -euo pipefail

TEAM_ACCOUNT="devlee328288"

if [ "$#" -eq 0 ]; then
  echo "사용법: bash scripts/gh-team.sh <gh 명령...>" >&2
  echo "예:     bash scripts/gh-team.sh pr create --fill" >&2
  exit 1
fi

# 금지 명령을 스크립트가 직접 막는다. 규약을 문서에만 두면 언젠가 누가 넘는다.
#
# ⚠️ **이 wrapper 를 거치지 않은 `gh` 는 못 막는다.** 2026-08-26 에 실제로 겪었다 —
#    PR 본문을 쓰면서 heredoc 구분자를 따옴표 없이(`<<PY`) 열었더니 본문 안의
#    백틱 `` `gh pr merge` `` 가 **셸 명령 치환으로 실행**됐다.
#    다행히 플래그(`--merge`/`--squash`)가 없어 실패했고 아무것도 머지되지 않았다.
#
#    교훈 둘:
#      · 문서·본문을 heredoc 으로 쓸 때는 **반드시 `<<'EOF'`** (따옴표) 를 쓴다.
#        변수를 넣어야 하면 파일로 먼저 쓰고 python 으로 치환한다.
#      · 이 가드는 **마지막 그물이지 유일한 그물이 아니다.**
for arg in "$@"; do
  if [ "$arg" = "merge" ]; then
    case " $* " in
      *" pr "*)
        echo "🚫 gh pr merge 는 이 저장소에서 금지입니다 (AGENTS.md 1.5 — 계정 정지 이력)." >&2
        echo "   머지는 GitHub 웹에서 사람이 직접 합니다." >&2
        exit 1
        ;;
    esac
  fi
done

if ! TOKEN="$(gh auth token --user "$TEAM_ACCOUNT" 2>/dev/null)"; then
  echo "❌ 팀 계정($TEAM_ACCOUNT) 토큰을 찾지 못했습니다." >&2
  echo "   먼저 한 번 로그인해 두세요:  gh auth login" >&2
  echo "   (로그인만 하면 됩니다. 활성 계정을 바꿀 필요는 없습니다.)" >&2
  exit 1
fi

GH_TOKEN="$TOKEN" exec gh "$@"
