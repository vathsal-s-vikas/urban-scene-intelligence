# PowerShell script to download CLIP model and embeddings
Write-Host "🔹 Downloading CLIP model and embeddings..." -ForegroundColor Blue

# Ensure Python environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Please activate your Python virtual environment first!" -ForegroundColor Red
    Write-Host "   Run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

# Install required packages if not already installed
Write-Host "🔹 Installing required packages..." -ForegroundColor Blue
pip install torch transformers requests tqdm

# Run the Python download script
Write-Host "🔹 Running CLIP model download script..." -ForegroundColor Blue
python scripts/download_clip.py

# Download embeddings if they don't exist
$embeddingsPath = "clip_attribute_embeddings.pt"
if (-not (Test-Path $embeddingsPath)) {
    Write-Host "🔹 Downloading CLIP embeddings..." -ForegroundColor Blue
    # Add the embeddings download URL here
    # $url = "YOUR_EMBEDDINGS_URL"
    # Invoke-WebRequest -Uri $url -OutFile $embeddingsPath
    Write-Host "⚠️ Please manually download the CLIP embeddings file and place it at: $embeddingsPath" -ForegroundColor Yellow
} else {
    Write-Host "✅ CLIP embeddings file already exists!" -ForegroundColor Green
}

Write-Host "✅ Setup complete!" -ForegroundColor Green