# Keep FastAPI on 127.0.0.1:8000. Intended for a Windows Scheduled Task so it
# survives editor/chat sessions. Loops forever; restarts uvicorn if /api/health fails.
$ErrorActionPreference = "Continue"
$Root = "C:\Users\Panshul\Desktop\zed-ps2-tracker-main"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Dir = Join-Path $env:USERPROFILE ".grok\long-running-background-tasks"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Log = Join-Path $Dir "dev_server_zed_ps2.log"
$Err = Join-Path $Dir "dev_server_zed_ps2.err.log"
$PidFile = Join-Path $Dir "zed_ps2_uvicorn.pid"
$Health = "http://127.0.0.1:8000/api/health"
$env:ZED_CORS_ORIGINS = "*"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $Root

function Test-Healthy {
  try {
    $r = Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 8
    return ($r.StatusCode -eq 200 -and $r.Content -match '"status"\s*:\s*"ok"')
  } catch {
    return $false
  }
}

function Stop-Listeners {
  if (Test-Path $PidFile) {
    $old = 0
    [void][int]::TryParse((Get-Content $PidFile | Select-Object -First 1), [ref]$old)
    if ($old -gt 0) { Stop-Process -Id $old -Force -ErrorAction SilentlyContinue }
  }
  Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 1
}

function Start-Server {
  Stop-Listeners
  if (-not (Test-Path $Py)) { throw "missing venv python: $Py" }
  $p = Start-Process -FilePath $Py -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $Log -RedirectStandardError $Err `
    -ArgumentList @("-m", "uvicorn", "serve.app:app", "--host", "127.0.0.1", "--port", "8000")
  Set-Content -Path $PidFile -Value $p.Id
}

while ($true) {
  if (-not (Test-Healthy)) {
    try { Start-Server } catch { Add-Content -Path $Err -Value ("$(Get-Date -Format o) start failed: " + $_.Exception.Message) }
  }
  Start-Sleep -Seconds 8
}
