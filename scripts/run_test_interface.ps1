param(
    [int]$Port = 8765,
    [string]$HostName = "127.0.0.1"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $projectRoot ".venv\Scripts\python.exe"),
    (Join-Path $projectRoot ".conda\python.exe")
)
$projectPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $projectPython) {
    throw "Không tìm thấy Python environment. Hãy tạo .venv hoặc .conda trong: $projectRoot"
}

Write-Host "Dùng environment: $projectPython"
& $projectPython (Join-Path $PSScriptRoot "test_interface.py") --host $HostName --port $Port
exit $LASTEXITCODE
