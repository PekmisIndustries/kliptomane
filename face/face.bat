
@echo off
setlocal enabledelayedexpansion

rem Get start time
for /F "tokens=1-4 delims=:.," %%a in ("%time%") do (
	set /A "start=(((%%a*60)+1%%b%%100)*60+1%%c%%100)*100+1%%d%%100"
)

python extract_frames.py
python detect_faces.py
python analyze_emotions.py
python aggregate_scores.py

rem Get end time
for /F "tokens=1-4 delims=:.," %%a in ("%time%") do (
	set /A "end=(((%%a*60)+1%%b%%100)*60+1%%c%%100)*100+1%%d%%100"
)

rem Calculate elapsed time
set /A elapsed=end-start
set /A hh=elapsed/360000
set /A mm=(elapsed-hh*360000)/6000
set /A ss=(elapsed-hh*360000-mm*6000)/100
set /A ms=elapsed-hh*360000-mm*6000-ss*100

echo Time taken: %hh%:%mm%:%ss%.%ms%

pause