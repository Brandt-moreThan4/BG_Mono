@echo off

REM Path to your conda installation (adjust if different)
set CONDAPATH=C:\Users\User\miniconda3
REM Environment name (replace with your actual environment name)
set ENVNAME=bg1

REM Activate environment and run script
call "%CONDAPATH%\Scripts\activate.bat" %ENVNAME%
REM Change to the directory where your script is located
cd "C:\Users\User\OneDrive\Desktop\Code\BG_Mono"
REM Run the Python script
python generate_report.py
call conda deactivate
pause
