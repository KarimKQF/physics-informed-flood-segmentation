$ErrorActionPreference = "Stop"

function Resolve-Gsutil {
    $command = Get-Command gsutil -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "$env:ProgramFiles\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
        "${env:ProgramFiles(x86)}\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
        "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd",
        "$env:APPDATA\gcloud\google-cloud-sdk\bin\gsutil.cmd"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $ProjectRoot "data\raw\Sen1Floods11"
$Bucket = "gs://sen1floods11"
$Gsutil = Resolve-Gsutil

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $Gsutil) {
    Write-Host "gsutil is not installed or not available in PATH."
    Write-Host "Install Google Cloud SDK first:"
    Write-Host "  https://cloud.google.com/sdk/docs/install"
    Write-Host ""
    Write-Host "After installation, restart PowerShell and verify with:"
    Write-Host "  gsutil version"
    Write-Host ""
    Write-Host "Then download with:"
    Write-Host "  gsutil -m rsync -r gs://sen1floods11 data/raw/Sen1Floods11"
    exit 1
}

if ($args.Count -gt 0 -and $args[0] -eq "--list") {
    Write-Host "Listing Sen1Floods11 bucket:"
    & $Gsutil ls $Bucket
    exit 0
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Output directory: $OutputDir"
Write-Host "Source bucket: $Bucket"
Write-Host "gsutil: $Gsutil"
Write-Host ""
Write-Host "This command uses rsync without deletion flags, so local extra files are not removed."
Write-Host "Starting download/synchronization..."

Set-Location $ProjectRoot
& $Gsutil -m rsync -r $Bucket "data/raw/Sen1Floods11"

Write-Host "Sen1Floods11 synchronization complete."
