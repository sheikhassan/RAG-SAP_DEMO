# Start the production Qdrant server for this project.
Set-Location $PSScriptRoot
docker compose up -d
docker compose ps
Write-Host ""
Write-Host "Qdrant REST:      http://127.0.0.1:6333"
Write-Host "Qdrant dashboard: http://127.0.0.1:6333/dashboard"
Write-Host "Then ingest:      .\.venv\Scripts\python.exe ingest.py --reset"
