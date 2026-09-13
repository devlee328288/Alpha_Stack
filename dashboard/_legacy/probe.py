# dashboard/probe.py
import app_streamlit_data as hf
from huggingface_hub import list_repo_files

df = hf.load_kospi200()
print("shape:", df.shape)
print("index:", df.index[:3], "→", df.index[-3:])
print("columns:")
for c in df.columns:
    print(f"  {c:30s} {str(df[c].dtype):10s}  "
          f"nan={df[c].isna().sum():5d}  "
          f"sample={df[c].dropna().iloc[0] if df[c].notna().any() else None}")

print("\n개별종목 개수:", len(hf.list_stocks()))
print("예시 코드:", hf.list_stocks()[:10])

files = list_repo_files(repo_id="qurious-quant/alphastack-krx-dev", repo_type="dataset")
print("\n=== HF 저장소 파일 목록 ===")
for f in files:
    print(" ", f)
