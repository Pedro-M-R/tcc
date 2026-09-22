@echo off
cd /d "%~dp0"
if exist ".venv-site\Scripts\python.exe" (
    ".venv-site\Scripts\python.exe" -m streamlit run web/app.py
) else (
    python -m streamlit run web/app.py
)
pause
