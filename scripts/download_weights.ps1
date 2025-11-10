# PowerShell script to download model weights used by the project.
# Usage:
#   .\scripts\download_weights.ps1 --yolo
#   .\scripts\download_weights.ps1 --all

param(
    [switch]$yolo,
    [switch]$flan
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Push-Location $root

if ($yolo) {
    Write-Host "Downloading YOLO weights (yolov8s.pt)..."
    $yoloUrl = "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8s.pt"
    $dest = Join-Path $root "yolov8s.pt"
    Invoke-WebRequest -Uri $yoloUrl -OutFile $dest
    Write-Host "Saved to $dest"
}

if ($flan) {
    Write-Host "Note: Flan-T5 models are usually downloaded at runtime via Hugging Face Transformers."
    Write-Host "To pre-download a model, run a small Python snippet that calls AutoModel.from_pretrained() and saves the cache."
}

if (-not $yolo -and -not $flan) {
    Write-Host "Usage: .\scripts\download_weights.ps1 --yolo | --flan | --all"
}

Pop-Location
