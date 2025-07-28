@echo off
echo Starting FastAPI server...
start cmd /k "cd /d %~dp0 && uvicorn app:app --host 0.0.0.0 --port 8000"

timeout /t 5 >nul

echo Starting ngrok tunnel...
start cmd /k "cd /d C:\ngrok && ngrok http 8000"

echo Done! FastAPI and Ngrok are now running.
pause
