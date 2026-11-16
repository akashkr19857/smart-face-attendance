@echo off
cd /d "%~dp0"
echo Starting Smart Gate Attendance - dashboard: http://localhost:8000
py -3.11 main.py %*
pause
