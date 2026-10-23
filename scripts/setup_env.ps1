# One-time environment setup on Windows (PowerShell), from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 [-Cpu] [-SkipMfa]
# Creates .venv (Python 3.11) with CUDA 12.6 torch, all requirements, spaCy en_core_web_sm (for
# ERRANT, evaluation only) and the editable `subfix` package; optionally the `mfa` conda env.
param([switch]$Cpu, [switch]$SkipMfa)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
$pip = @('-m', 'pip', 'install', '--retries', '10', '--timeout', '60')
& $py -m pip install --upgrade pip setuptools wheel

# 1. torch (resumable download of the 2.6 GB CUDA wheel; flaky networks reset long transfers)
if ($Cpu) {
    & $py @pip torch
} else {
    $whl = 'wheels\torch-2.14.1+cu126-cp311-cp311-win_amd64.whl'
    New-Item -ItemType Directory -Force wheels | Out-Null
    & .\scripts\fetch_resumable.ps1 -Url 'https://download.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp311-cp311-win_amd64.whl' -Out $whl -Size 2602753280
    & $py @pip $whl
}

# 2. requirements
& $py @pip -r requirements.txt

# 3. spaCy model (ERRANT evaluation only) + this package
& $py -m spacy download en_core_web_sm
& $py @pip --no-deps -e .
& $py -c "import torch, ctranslate2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ctranslate2 cuda devices', ctranslate2.get_cuda_device_count())"

# 4. Montreal Forced Aligner in its own conda env (silver-gold timing references, evaluation only)
if (-not $SkipMfa) {
    $conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if (-not $conda) { $conda = "$env:USERPROFILE\Miniconda3\condabin\conda.bat" }
    $env:CONDA_REMOTE_MAX_RETRIES = '10'
    & $conda env create -f environment-mfa.yml
    & $conda run -n mfa mfa model download acoustic english_us_arpa
    & $conda run -n mfa mfa model download dictionary english_us_arpa
}

# 5. Whisper models (~2.5 GB) + check of our self-trained models: python scripts\prefetch_models.py
Write-Output "Done. Next: .venv\Scripts\python scripts\prefetch_models.py ; .venv\Scripts\python -m pytest"
