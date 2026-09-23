param(
    [string]$Python = "python",
    [string]$BuildTag = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dist = Join-Path $Root "dist\$BuildTag"
$Work = Join-Path $Root "build\$BuildTag"
$CommonExcludes = @(
    "--exclude-module", "torch",
    "--exclude-module", "matplotlib",
    "--exclude-module", "IPython",
    "--exclude-module", "jupyter",
    "--exclude-module", "notebook"
)

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $Work | Out-Null
    & $Python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name StallWarning `
        --specpath $Work --distpath $Dist --workpath $Work `
        @CommonExcludes `
        --add-data "$(Join-Path $Root 'assets');assets" `
        --add-data "$(Join-Path $Root 'config');config" `
        (Join-Path $Root "app.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name SensorSimulator `
        --specpath $Work --distpath $Dist --workpath $Work `
        @CommonExcludes `
        --add-data "$(Join-Path $Root 'assets');assets" `
        --add-data "$(Join-Path $Root 'config');config" `
        (Join-Path $Root "sensor_simulator_gui.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $Workspace = Split-Path -Parent $Root
    Copy-Item -LiteralPath (Join-Path $Workspace "data\pressure_timestep.csv") -Destination $Dist -Force
    Copy-Item -LiteralPath (Join-Path $Workspace "data\pressure_u1_initialization.csv") -Destination $Dist -Force
    Copy-Item -LiteralPath (Join-Path $Root "run_simulator.ps1") -Destination $Dist -Force
    Write-Host "Build output: $Dist"
    exit 0
}
finally {
    Pop-Location
}
