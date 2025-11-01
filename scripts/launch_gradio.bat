@echo off
REM Script de lancement de l'interface Gradio pour la génération 3D

echo ========================================
echo Interface Gradio - Generateur 3D
echo ========================================
echo.

REM Se déplacer vers la racine du projet
cd /d "%~dp0\.."

REM Activer l'environnement virtuel si disponible
if exist "venv\Scripts\activate.bat" (
    echo Activation de l'environnement virtuel...
    call venv\Scripts\activate.bat
) else (
    echo Aucun environnement virtuel trouve
)

echo.
echo Lancement de l'interface...
echo.

python -m src.interface.gradio_app

pause
