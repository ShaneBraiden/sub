# Full evaluation sequence (run after tools\build_benchmark.py and experiments\tune_fusion.py):
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_experiments.ps1 [-Fresh] [-SkipMain] [-SkipAblations]
# Logs go to cache\logs\*.log; tables to results\.
param([switch]$Fresh, [switch]$SkipMain, [switch]$SkipAblations, [switch]$SkipSweep,
      [string[]]$Ablations = @('timing_source', 'segmenter', 'audio_guard', 'fusion_signals', 'asr_size', 'gec_model'))
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force cache\logs | Out-Null
if ($Fresh) {   # cached system outputs / text stages were produced by older code
    Remove-Item -Recurse -Force experiments\outputs, cache\text -ErrorAction SilentlyContinue
}
function Step($name, [string[]]$argv) {
    $t = Get-Date
    Write-Output ("[{0}] start {1}" -f $t.ToString('HH:mm'), $name)
    & $py @argv *> "cache\logs\$name.log"
    Write-Output ("[{0}] done  {1} ({2:N0} min, exit {3})" -f (Get-Date).ToString('HH:mm'), $name, ((Get-Date) - $t).TotalMinutes, $LASTEXITCODE)
}
if (-not $SkipMain) {
    Step 'main' @('experiments\run_eval.py', '--manifest', 'test', '--systems', 'noop', 'b1_symspell_lt', 'b2_ffsubsync',
                  'b3_raw_whisper', 'ours', 'modeb_raw', 'modeb_ours')
}
if (-not $SkipAblations) {
    foreach ($abl in $Ablations) {
        Step "ablation_$abl" @('experiments\run_eval.py', '--manifest', 'test', '--ablation', $abl,
                               '--datasets', 'libri-test-clean', 'ted-test', '--no-errant')
    }
}
if (-not $SkipSweep) {
    Step 'sweep' @('experiments\run_eval.py', '--sweep', '0.05', '0.10', '0.20')
}
Step 'plots' @('experiments\plots.py')
Step 'sync_eval' @('experiments\sync_eval.py', '--manifest', 'test')
Step 'error_analysis' @('experiments\error_analysis.py', '--manifest', 'test')
Step 'summarize' @('experiments\summarize.py')
