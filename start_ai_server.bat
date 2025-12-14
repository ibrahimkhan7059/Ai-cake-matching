@echo off
echo Starting AI Cake Matching Server...
echo Server will run on: http://192.168.100.4:5000/api/v1/ai-cake
echo.

cd "c:\xampp\htdocs\FYP\Ai-matching system"
python cake_api.py

pause
