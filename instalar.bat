@echo off

echo ================================
echo Instalando dependencias Control4
echo ================================

python --version
if errorlevel 1 (
    echo Error: Python 3 no esta instalado.
    pause
    exit /b 1
)

python -m venv venv

call venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Instalacion completada.
echo Para ejecutar:
echo venv\Scripts\activate
echo python "codigo pregunta 4.py"

pause
