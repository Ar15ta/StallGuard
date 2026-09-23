param(
    [string]$Csv = "pressure_timestep.csv",
    [double]$StartStep = 88000,
    [double]$Rate = 3000,
    [int]$BatchSize = 50,
    [int]$Port = 8765,
    [string]$HostName = "127.0.0.1",
    [int]$MaxRows = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Executable = Join-Path $Root "SensorSimulator\SensorSimulator.exe"
$CsvPath = Join-Path $Root $Csv
if (-not (Test-Path -LiteralPath $Executable)) { throw "找不到传感器模拟器：$Executable" }
if (-not (Test-Path -LiteralPath $CsvPath)) { throw "找不到数据文件：$CsvPath" }

$Arguments = @($CsvPath, "--host", $HostName, "--port", $Port, "--rate", $Rate, "--batch-size", $BatchSize, "--start-step", $StartStep)
if ($MaxRows -gt 0) { $Arguments += @("--max-rows", $MaxRows) }
& $Executable @Arguments
exit $LASTEXITCODE
