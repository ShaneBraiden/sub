# Start the web app: builds the frontend when needed, then runs the backend on http://127.0.0.1:8000
#   powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1            # opens the browser
#   powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1 -Rebuild   # force a frontend rebuild
#   powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1 -Dev       # Vite dev server (:5173) + backend with reload
param(
    [int]$Port = 8000,
    [switch]$Rebuild,
    [switch]$Dev,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$frontend = Join-Path $root "frontend"
if (-not (Test-Path $python)) { throw "No .venv found - run scripts\setup_env.ps1 first." }

Push-Location $frontend
try {
    if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
        Write-Host "Installing frontend packages (npm install)..."
        npm install --no-audit --no-fund --fetch-retries=6
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }
    if ($Dev) {
        Write-Host "Frontend dev server: http://localhost:5173 (API proxied to :$Port)"
        Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory $frontend
    } elseif ($Rebuild -or -not (Test-Path (Join-Path $frontend "dist\index.html"))) {
        Write-Host "Building the frontend..."
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
    }
} finally {
    Pop-Location
}

Push-Location $root
try {
    $serverArgs = @("-m", "app.backend", "--port", $Port)
    if ($Dev) { $serverArgs += @("--reload", "--no-browser") }
    elseif ($NoBrowser) { $serverArgs += "--no-browser" }
    & $python @serverArgs
} finally {
    Pop-Location
}
