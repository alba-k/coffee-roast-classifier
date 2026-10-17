# Creates the virtual environment and installs the dependencies.
#   powershell -ExecutionPolicy Bypass -File setup.ps1
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
