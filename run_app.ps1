# Run Streamlit with the project venv (avoids "ModuleNotFoundError" from system Python).
Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\app.py" --server.fileWatcherType none @args
