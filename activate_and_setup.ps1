Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"
Write-Host "Running GPU setup..." -ForegroundColor Green
python setup_gpu_env.py
Read-Host "Press Enter to continue..."