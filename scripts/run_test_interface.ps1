param(
    [int]$Port = 8765,
    [string]$HostName = "127.0.0.1"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot ".conda\python.exe"

if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Không tìm thấy Python environment của project: $projectPython"
}

Write-Host "Dùng environment: $projectPython"
& $projectPython (Join-Path $PSScriptRoot "test_interface.py") --host $HostName --port $Port
exit $LASTEXITCODE
