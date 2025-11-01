@echo off
REM Script pour tester les orientations d'un modèle 3D
echo ========================================
echo Test des orientations de modele 3D
echo ========================================
echo.

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Demander le chemin du fichier
set /p MODEL_PATH="Entrez le chemin du modele 3D (ou glissez-deposez le fichier): "

REM Retirer les guillemets si présents
set MODEL_PATH=%MODEL_PATH:"=%

REM Vérifier que le fichier existe
if not exist "%MODEL_PATH%" (
    echo.
    echo ERREUR: Le fichier n'existe pas !
    echo.
    pause
    exit /b 1
)

echo.
echo Generation des orientations...
echo.

REM Lancer le script Python
python scripts\test_orientations.py "%MODEL_PATH%"

echo.
echo ========================================
echo Termine !
echo ========================================
echo.
pause
