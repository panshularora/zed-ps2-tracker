# Register logon tasks that keep the API and public tunnel running outside any chat session.
$ErrorActionPreference = "Stop"
$Root = "C:\Users\Panshul\Desktop\zed-ps2-tracker-main"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -StartWhenAvailable

function Install-Keeper([string]$Name, [string]$File) {
  $script = Join-Path $Root "scripts\windows\$File"
  $action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings `
    -Description "Zed PS-2: keep FastAPI/tunnel running" -Force | Out-Null
  Start-ScheduledTask -TaskName $Name
}

Install-Keeper "ZedPS2Backend" "keep_backend.ps1"
Install-Keeper "ZedPS2Tunnel" "keep_tunnel.ps1"
Write-Output "installed ZedPS2Backend and ZedPS2Tunnel"
