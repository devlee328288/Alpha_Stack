# dashboard/probe2.py
import json

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "qurious-quant/alphastack-krx-dev"

def fetch(path):
    return hf_hub_download(repo_id=REPO, filename=path, repo_type="dataset")

# ── 1) sample_codes.json ────────────────────────────────────
print("=" * 60)
print("small/sample_codes.json")
print("=" * 60)
with open(fetch("small/sample_codes.json"), encoding="utf-8") as f:
    codes = json.load(f)
print("type:", type(codes))
print("값 예시:", str(codes)[:400])

# ── 2) features_labels_stocks30_dev.csv ─────────────────────
print("\n" + "=" * 60)
print("small/features_labels_stocks30_dev.csv")
print("=" * 60)
df = pd.read_csv(fetch("small/features_labels_stocks30_dev.csv"), nrows=10)
print("columns:", list(df.columns))
print(df.head(3).to_string())
# 종목 식별 컬럼 후보 찾기
for cand in ["ticker", "code", "stock_code", "symbol", "종목코드"]:
    if cand in df.columns:
        print(f"\n>>> 종목 식별 컬럼 발견: '{cand}'")
        full = pd.read_csv(fetch("small/features_labels_stocks30_dev.csv"),
                           usecols=[cand])
        print(f"    유니크 값 개수: {full[cand].nunique()}")
        print(f"    예시: {sorted(full[cand].astype(str).unique())[:10]}")
        break
else:
    print("\n>>> 종목 식별 컬럼을 못 찾음. 위 columns 를 보고 알려주세요.")

# ── 3) stocks_sample30_train_dev.csv ────────────────────────
print("\n" + "=" * 60)
print("small/stocks_sample30_train_dev.csv")
print("=" * 60)
df2 = pd.read_csv(fetch("small/stocks_sample30_train_dev.csv"), nrows=5)
print("columns:", list(df2.columns))
print(df2.head(3).to_string())

# ── 4) 샘플코드 하나로 필터 테스트 ───────────────────────────
print("\n" + "=" * 60)
print("종목 필터 테스트")
print("=" * 60)
try:
    if isinstance(codes, dict):
        sample_code = list(codes.keys())[0]
    elif isinstance(codes, list):
        sample_code = codes[0] if isinstance(codes[0], str) else list(codes[0].values())[0]
    else:
        sample_code = None
    print("샘플 코드:", sample_code)

    for col in ["ticker", "code", "stock_code", "symbol", "종목코드"]:
        try:
            df_full = pd.read_csv(fetch("small/features_labels_stocks30_dev.csv"),
                                  dtype={col: str})
            sub = df_full[df_full[col] == str(sample_code)]
            if len(sub) > 0:
                print(f"  '{col}' == {sample_code} → {len(sub)}행 매칭됨")
                print("  컬럼:", list(sub.columns))
                break
        except Exception:
            continue
except Exception as e:
    print("테스트 스킵:", e)
