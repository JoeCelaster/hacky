@echo off
echo ============================================================
echo   Medical RAG Chatbot - Installation and Testing
echo ============================================================
echo.

echo [1/3] Installing dependencies...
py quick_install.py
if %ERRORLEVEL% NEQ 0 (
    echo Installation encountered issues but continuing...
)

echo.
echo [2/3] Testing system...
py test_quick.py
if %ERRORLEVEL% NEQ 0 (
    echo Some tests failed but continuing...
)

echo.
echo [3/3] Ready to start server!
echo.
echo To start the server, run: py server.py
echo Then open: http://localhost:8000/docs
echo.
pause



