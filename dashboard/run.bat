@echo off
cd /d %~dp0
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
streamlit run app.py