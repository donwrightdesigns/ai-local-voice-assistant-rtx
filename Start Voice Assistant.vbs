' Voice Assistant Launcher - Double-click to start
' This script launches the PowerShell script without showing a console window

Set objShell = CreateObject("WScript.Shell")

' Get the directory where this script is located
strScriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Change to the script directory and run PowerShell script
objShell.CurrentDirectory = strScriptDir
objShell.Run "powershell.exe -ExecutionPolicy Bypass -File ""start_voice_assistant.ps1""", 1, False

' Show a brief notification
Set objNotify = CreateObject("WScript.Shell")
objNotify.Popup "🎤 Voice Assistant RTX is starting..." & vbCrLf & vbCrLf & "A terminal window will open shortly.", 2, "Voice Assistant RTX", 64
