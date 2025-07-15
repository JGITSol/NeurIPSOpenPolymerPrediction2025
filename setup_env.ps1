# PowerShell script to set up the development environment on Windows

Write-Host "Setting up Polymer Prediction development environment..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install the package with development dependencies
Write-Host "Installing package with development dependencies..." -ForegroundColor Yellow
pip install -e ".[dev,docs]"

# Install pre-commit hooks
Write-Host "Installing pre-commit hooks..." -ForegroundColor Yellow
pre-commit install

# Create necessary directories
Write-Host "Creating project directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data", "logs", "outputs", "runs"

# Generate sample data
Write-Host "Generating sample data..." -ForegroundColor Yellow
python scripts/setup_data.py --n_samples 100 --split

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To activate the environment in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To run tests: make test" -ForegroundColor Cyan
Write-Host "To train a model: make train" -ForegroundColor Cyan