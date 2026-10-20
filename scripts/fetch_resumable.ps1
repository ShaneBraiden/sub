# Resumable download with retries (the network here resets long transfers).
# Usage: .\scripts\fetch_resumable.ps1 -Url <url> -Out <file> [-Size <expected bytes>] [-MaxTries 200]
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Out,
    [long]$Size = 0,
    [int]$MaxTries = 200
)
$ErrorActionPreference = 'Continue'
if ($Size -le 0) {
    $head = curl.exe -sIL $Url | Select-String -Pattern '^[Cc]ontent-[Ll]ength:\s*(\d+)' | Select-Object -Last 1
    if ($head) { $Size = [long]$head.Matches[0].Groups[1].Value }
}
for ($try = 1; $try -le $MaxTries; $try++) {
    $have = 0
    if (Test-Path $Out) { $have = (Get-Item $Out).Length }
    if ($Size -gt 0 -and $have -ge $Size) { Write-Output "DONE $Out ($have bytes)"; exit 0 }
    Write-Output ("try {0}: have {1:N0} / {2:N0} bytes" -f $try, $have, $Size)
    curl.exe -L --fail -s -S --connect-timeout 30 --speed-time 60 --speed-limit 1024 -C - -o $Out $Url
    if ($LASTEXITCODE -eq 0 -and $Size -le 0) { Write-Output "DONE $Out"; exit 0 }
    Start-Sleep -Seconds 3
}
Write-Output "FAILED $Out"
exit 1
