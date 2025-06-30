# PowerShell script to set up virtual environment and install dependencies

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
. .\venv\Scripts\Activate.ps1

# Install uv
Write-Host "Installing uv..." -ForegroundColor Green
pip install uv

# Install dependencies using uv
Write-Host "Installing dependencies using uv..." -ForegroundColor Green
uv pip install -r requirements.txt

# Install the package in development mode
Write-Host "Installing package in development mode..." -ForegroundColor Green
pip install -e .

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "To activate the virtual environment in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
