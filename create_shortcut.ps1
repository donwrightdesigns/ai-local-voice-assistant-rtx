# Create Desktop Shortcut for Voice Assistant
Write-Host "Creating Desktop Shortcut..." -ForegroundColor Cyan
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VBSPath = Join-Path $ScriptDir "Start Voice Assistant.vbs"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Voice Assistant RTX.lnk"
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $VBSPath
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.Description = "Voice Assistant RTX"
$Shortcut.IconLocation = "C:\Windows\System32\Speech\SpeechUX\sapi.cpl,0"
$Shortcut.Save()
Write-Host "Shortcut created on Desktop!" -ForegroundColor Green
