# Start Billie for local development: FastAPI on :8000 and the React dev server on :5173.
# Usage:  .\scripts\dev.ps1            (first run installs everything)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating Python 3.12 virtual environment..." -ForegroundColor Cyan
    py -3.12 -m venv (Join-Path $root "backend\.venv")
    & $py -m pip install --upgrade pip
    & $py -m pip install -e "$root\backend[dev]"
}
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location (Join-Path $root "frontend"); npm install; Pop-Location
}
try { Invoke-RestMethod http://127.0.0.1:11434/api/version | Out-Null }
catch { Write-Warning "Ollama is not running. Start it (or install from https://ollama.com) for AI answers." }

$api = Start-Process -PassThru -NoNewWindow -FilePath $py -WorkingDirectory (Join-Path $root "backend") `
    -ArgumentList "-m", "uvicorn", "billie.api.app:create_app", "--factory", "--reload", "--port", "8000"
Write-Host "API      http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Billie   http://127.0.0.1:5173" -ForegroundColor Green
try {
    Push-Location (Join-Path $root "frontend"); npm run dev
} finally {
    Pop-Location; Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
}
