@echo off
echo Restarting server and ingesting documents...
echo.
echo [1/2] Stopping any running servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq py server.py*" >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/2] Starting fresh server...
start /B py server.py

echo Waiting for server to start...
timeout /t 10 /nobreak

echo.
echo Running ingestion...
py ingest_now.py



