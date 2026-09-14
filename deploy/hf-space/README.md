---
title: Alphastack Qurious
emoji: 🏃
colorFrom: yellow
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AlphaStack 대시보드 (팀 Qurious)

주가지수·개별종목의 5일 뒤 등락 방향 예측과 성과 검증 대시보드입니다.

- 코드: https://github.com/devlee328288/Alpha_Stack (`dashboard/`)
- 이 Space 에는 `Dockerfile` 과 이 README 만 있습니다. 앱 코드는 빌드할 때 GitHub `main` 에서 받습니다.
- 정본: 저장소 `deploy/hf-space/` — Space 파일을 고칠 때는 저장소부터 고칩니다.
- 새 코드를 반영하려면 Settings → **Factory rebuild**.
- 필요한 Secret: `HF_TOKEN` — 데이터셋 `qurious-quant/alphastack-krx-dev` 읽기 권한.
- 화면의 RUN 버튼은 서버에서 학습을 돌립니다. 같은 설정의 결과는 앱이 재시작되기 전까지 캐시됩니다.

> ⚠️ 저장소 안에서 이 파일 맨 위의 `---` 머리말은 **Space 설정**입니다(Hugging Face 가 읽음). 지우지 마세요.
