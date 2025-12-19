@echo off
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo Installing roboflow without broken dependencies...
pip install roboflow --no-deps
if %errorlevel% neq 0 exit /b %errorlevel%

echo Installation complete!
pause
