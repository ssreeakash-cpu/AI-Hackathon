# Stop any hanging uvicorn processes
Stop-Process -Name "uvicorn" -ErrorAction Ignore

# Start Backend in the background on Port 8001 (watching only source folders)
Start-Process -FilePath ".\.venv\Scripts\uvicorn.exe" -ArgumentList "backend.main:app", "--host", "127.0.0.1", "--port", "8001", "--reload", "--reload-dir", "backend", "--reload-dir", "inference", "--reload-dir", "models", "--reload-dir", "preprocessing" -NoNewWindow

# Start Frontend
Set-Location -Path "frontend"
npm run dev
