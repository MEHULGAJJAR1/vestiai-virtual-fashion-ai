@echo off
REM =====================================================================================
REM VestiAI - one-command training runner (Windows)
REM
REM   scripts\run_training.bat                            QUICK_DEMO on the sample dataset
REM   scripts\run_training.bat --mode FINE_TUNE --dataset datasets\viton_hd
REM   scripts\run_training.bat --help
REM
REM Everything is delegated to the Python CLI tools, so you can always run the same steps
REM by hand:
REM
REM   python scripts\setup.py --profile ml
REM   python scripts\prepare_dataset.py --use-samples
REM   python scripts\validate_dataset.py --dataset datasets\samples
REM   python scripts\train.py --mode QUICK_DEMO
REM
REM On macOS/Linux use scripts/run_training.sh instead.
REM =====================================================================================
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." || (echo Cannot enter the project folder & exit /b 2)

set "PY=python"
if defined PYTHON set "PY=%PYTHON%"

set "MODE=QUICK_DEMO"
set "DATASET="
set "EXTRA="

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--mode"    ( set "MODE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--dataset" ( set "DATASET=%~2" & shift & shift & goto parse )
if /I "%~1"=="-h"        goto usage
if /I "%~1"=="--help"    goto usage
set "EXTRA=!EXTRA! %~1"
shift
goto parse

:usage
echo Usage: scripts\run_training.bat [--mode QUICK_DEMO^|FINE_TUNE^|FULL_TRAINING] [--dataset NAME^|PATH] [extra train.py flags]
popd
exit /b 0

:parsed
echo.
echo === VestiAI training - mode %MODE% ===

REM ---------------------------------------------------------------- 1. environment
echo.
echo [1/5] environment
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo     PyTorch is not installed - installing the ML profile ^(this can take a while^)...
    "%PY%" scripts\setup.py --profile ml
    if errorlevel 1 (
        echo X   ML profile install failed. Run it manually to see the full log:
        echo       %PY% scripts\setup.py --profile ml
        popd & exit /b 3
    )
) else (
    "%PY%" scripts\setup.py --check --profile ml
)

REM ---------------------------------------------------------------- 2. dataset
echo.
echo [2/5] dataset
if not "%DATASET%"=="" (
    "%PY%" scripts\validate_dataset.py --dataset "%DATASET%"
    if errorlevel 1 (
        echo X   dataset '%DATASET%' did not pass validation.
        popd & exit /b 4
    )
    goto dataset_done
)
if exist "datasets\samples\train\pairs.txt" (
    echo     using datasets\samples ^(already prepared^)
    "%PY%" scripts\validate_dataset.py --dataset datasets\samples
) else (
    echo     no dataset found - generating the sample dataset
    "%PY%" scripts\prepare_dataset.py --use-samples
)
:dataset_done

REM ---------------------------------------------------------------- 3. GPU check
echo.
echo [3/5] GPU check
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo     No CUDA GPU - QUICK_DEMO still runs on CPU ^(a few minutes^).
    echo     FINE_TUNE / FULL_TRAINING belong on a CUDA box; see configs\training_colab.yaml.
) else (
    echo     CUDA available.
)

REM ---------------------------------------------------------------- 4. training
echo.
echo [4/5] training ^(mode=%MODE%^)
set "TRAIN_ARGS=--mode %MODE%"
if not "%DATASET%"=="" set "TRAIN_ARGS=%TRAIN_ARGS% --dataset %DATASET%"
"%PY%" scripts\train.py %TRAIN_ARGS% %EXTRA%
if errorlevel 1 (
    echo X   training exited with an error - see the traceback above.
    popd & exit /b 5
)

REM ---------------------------------------------------------------- 5. next steps
echo.
echo [5/5] next steps
if exist "checkpoints\best_model"   (echo     best   checkpoint: checkpoints\best_model)   else (echo     best   checkpoint: missing)
if exist "checkpoints\latest_model" (echo     latest checkpoint: checkpoints\latest_model) else (echo     latest checkpoint: missing)
echo     validate : python scripts\validate.py --checkpoint checkpoints\best_model
echo     evaluate : python scripts\evaluate.py --checkpoint checkpoints\best_model
echo     serve    : python run.py            ^(Model Status - Reload checkpoint^)

echo.
echo Training run finished.
popd
endlocal
