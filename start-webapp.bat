@echo off
REM ============================================================
REM  OCGAN - Avvio tutto-in-uno della webapp (un solo comando)
REM  Il backend FastAPI serve anche il frontend gia' compilato
REM  (app\frontend\dist) su http://localhost:8000
REM ============================================================
title OCGAN webapp
cd /d "D:\OCGAN\app\ocgan-modernized"

if not exist ".venv\Scripts\python.exe" (
  echo [ERRORE] Ambiente Python non trovato in .venv
  echo Esegui prima il setup una tantum ^(vedi D:\OCGAN\App.txt^):
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install torch torchvision
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-webapp.txt
  echo.
  pause
  exit /b 1
)

echo.
echo  Avvio server su http://localhost:8000
echo  Attendi il messaggio "Application startup complete", poi apri quell'indirizzo nel browser.
echo  Premi CTRL+C in questa finestra per fermare la webapp.
echo.
".venv\Scripts\python.exe" server.py --port 8000 --device auto

echo.
echo  Server terminato.
pause
