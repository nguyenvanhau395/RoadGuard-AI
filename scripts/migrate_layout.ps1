$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
function Resolve-ProjectPath([string]$relative) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside project: $relative"
    }
    return $resolved
}
function Get-Inventory([string]$path) {
    $item = Get-Item -LiteralPath $path
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $path -File -Recurse -Force) } else { @($item) }
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    return @{ files=$files.Count; bytes=[long]$bytes }
}
$pairs = [Collections.Generic.List[object]]::new()
@(
    @('crack-4\runs','archive\runs\segmentation\crack-4'),
    @('crack-4\hard_negative_from_prenn','archive\data\hard_negative_from_prenn'),
    @('crack-4','data\raw\crack-4'),
    @('outputs_42k','archive\runs\detection\outputs_42k'),
    @('det\test_image\output_image','archive\predictions\detection\images'),
    @('det\video_test\output_video','archive\predictions\detection\videos'),
    @('Seg\images\output_images','archive\predictions\segmentation\images'),
    @('Seg\videos\output_video','archive\predictions\segmentation\videos'),
    @('det','assets\detection'),
    @('Seg','assets\segmentation')
) | ForEach-Object { $pairs.Add($_) }
Get-ChildItem -LiteralPath $projectRoot -File -Filter '*.ipynb' | ForEach-Object { $pairs.Add(@($_.Name,('archive\notebooks\' + $_.Name))) }
Get-ChildItem -LiteralPath $projectRoot -File -Filter '*.pt' | ForEach-Object { $pairs.Add(@($_.Name,('models\pretrained\' + $_.Name))) }
Get-ChildItem -LiteralPath $projectRoot -File -Filter '*.zip' | ForEach-Object { $pairs.Add(@($_.Name,('data\archives\' + $_.Name))) }
Get-ChildItem -LiteralPath $projectRoot -File -Filter '*.yaml' | ForEach-Object { $pairs.Add(@($_.Name,('configs\architectures\' + $_.Name))) }
foreach ($name in @('BaoCao_RDD2022_YOLOv8m.docx','rdd2022_analysis.png')) { $pairs.Add(@($name,('archive\reports\' + $name))) }
$logPath = Resolve-ProjectPath 'reports\migration.json'
New-Item -ItemType Directory -Path (Split-Path $logPath) -Force | Out-Null
$records = [Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath $logPath) { @(Get-Content -LiteralPath $logPath -Raw -Encoding UTF8 | ConvertFrom-Json) | ForEach-Object { $records.Add($_) } }
foreach ($pair in $pairs) {
    $source = Resolve-ProjectPath $pair[0]
    $destination = Resolve-ProjectPath $pair[1]
    if (-not (Test-Path -LiteralPath $source)) { continue }
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    $before = Get-Inventory $source
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $destination
    $after = Get-Inventory $destination
    if ($before.files -ne $after.files -or $before.bytes -ne $after.bytes) { throw "Inventory mismatch: $destination" }
    $records.Add([ordered]@{source=$pair[0];destination=$pair[1];files=$after.files;bytes=$after.bytes;verified=$true;completed_at=(Get-Date -Format o)})
    ConvertTo-Json -InputObject @($records.ToArray()) -Depth 6 | Set-Content -LiteralPath $logPath -Encoding UTF8
    Write-Output "Moved $($pair[0]) -> $($pair[1]) ($($after.files) files)"
}
