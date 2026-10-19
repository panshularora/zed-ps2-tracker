# Public Cloudflare quick tunnel to the local API. Restarts if cloudflared dies.
# Writes the current URL to %USERPROFILE%\.grok\long-running-background-tasks\zed_ps2_tunnel.url
$ErrorActionPreference = "Continue"
$Dir = Join-Path $env:USERPROFILE ".grok\long-running-background-tasks"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Log = Join-Path $Dir "zed_ps2_tunnel.log"
$Err = Join-Path $Dir "zed_ps2_tunnel.err.log"
$UrlFile = Join-Path $Dir "zed_ps2_tunnel.url"
$Cf = "C:\Program Files (x86)\cloudflared\cloudflared.exe"

function Save-Url {
  $files = @($Log, $Err) | Where-Object { Test-Path $_ }
  if (-not $files) { return }
  $m = Select-String -Path $files -Pattern "https://[a-z0-9-]+\.trycloudflare.com" -ErrorAction SilentlyContinue |
    Select-Object -Last 1
  if ($m -and $m.Matches.Count -gt 0) {
    Set-Content -Path $UrlFile -Value $m.Matches[0].Value
  }
}

function Ensure-Tunnel {
  $running = Get-Process -Name cloudflared -ErrorAction SilentlyContinue
  if ($running) { Save-Url; return }
  if (Test-Path $Log) { Clear-Content $Log -ErrorAction SilentlyContinue }
  Start-Process -FilePath $Cf -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $Log -RedirectStandardError $Err `
    -ArgumentList @("tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate") | Out-Null
}

while ($true) {
  try { Ensure-Tunnel } catch { Add-Content -Path $Log -Value ("$(Get-Date -Format o) " + $_.Exception.Message) }
  Save-Url
  Start-Sleep -Seconds 8
}
