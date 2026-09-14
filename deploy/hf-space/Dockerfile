# AlphaStack 대시보드 — Hugging Face Space (Docker SDK)
#
# 이 파일의 정본은 저장소의 deploy/hf-space/Dockerfile 이다. Space 에는 같은 내용을 올린다.
# Space 에는 이 파일과 README.md 둘만 둔다. 앱 코드는 빌드할 때 GitHub 공개 저장소에서 받는다.
# 새 main 을 반영하려면 Space Settings → "Factory rebuild" (일반 Restart 는 git clone 층을 다시 안 돌 수 있다).
#
# 런타임에 필요한 Secret: HF_TOKEN (qurious-quant/alphastack-krx-dev 읽기 권한)
#   hf_hub_download 는 HF_TOKEN 이라는 이름만 읽는다 — 다른 이름이면 401.

FROM python:3.12-slim

# lightgbm 이 OpenMP(libgomp) 를 요구한다 · git 은 저장소를 받는 데 쓴다
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces 는 UID 1000 으로 실행한다 — 쓰기 권한이 필요한 캐시를 사용자 홈에 둔다
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8
WORKDIR /home/user

ARG REPO=https://github.com/devlee328288/Alpha_Stack.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} app

WORKDIR /home/user/app
# 대시보드 요구사항 — torch 는 파일 안의 --find-links 로 CPU 전용 wheel 을 받는다 (PR #266)
RUN pip install --no-cache-dir --user -r dashboard/requirements.txt

EXPOSE 7860
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
