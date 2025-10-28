# 🚀 Voice Assistant RTX Launcher Setup

## Quick Start - Create Desktop Icon

Run this **once** to create a clickable desktop shortcut:

```powershell
.\create_shortcut.ps1
```

This creates a **"Voice Assistant RTX.lnk"** icon on your desktop with a microphone icon.

## Usage

### Double-Click Desktop Icon
Just double-click the **Voice Assistant** icon on your desktop!

The launcher will:
1. ✅ Auto-start Ollama (if not running)
2. ✅ Activate conda environment
3. ✅ Launch voice assistant with Kokoro TTS
4. 🎤 Ready to use!

### Manual Launch
If you prefer terminal:
```powershell
.\start_voice_assistant.ps1
```

## Files Explained

| File | Purpose |
|------|---------|
| `Start Voice Assistant.vbs` | VBScript wrapper for double-click |
| `start_voice_assistant.ps1` | Main launcher (auto-starts Ollama) |
| `create_shortcut.ps1` | Creates desktop shortcut (run once) |

## Features

### Auto-Start Ollama
The launcher automatically checks if Ollama is running and starts it if needed:
- ✅ Detects if Ollama is already running
- 🚀 Starts Ollama in background if not running  
- ⏱️ Waits 3 seconds for Ollama to initialize
- 🎯 Then launches voice assistant

### Conda Environment
Automatically activates the `voice-assistant` conda environment:
- Python 3.12
- PyTorch with CUDA
- Kokoro TTS
- All dependencies

### No Console Window (VBS)
The `.vbs` launcher shows a brief popup then opens the terminal in the background.

## Pinning to Taskbar

1. Right-click the desktop shortcut
2. Select **"Pin to taskbar"**
3. Now you have one-click access! 📌

## Pinning to Start Menu

1. Right-click the desktop shortcut
2. Select **"Pin to Start"**
3. Access from Windows Start menu! 

## Custom Icon (Optional)

To use a custom icon:

1. Find a `.ico` file (microphone, robot, etc.)
2. Edit `create_shortcut.ps1` line 19:
   ```powershell
   $Shortcut.IconLocation = "C:\Path\To\Your\Icon.ico"
   ```
3. Run `.\create_shortcut.ps1` again

## Troubleshooting

### "Execution Policy" Error
Run this once in admin PowerShell:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Ollama Won't Start
The launcher tries to auto-start Ollama. If it fails:
1. Open a separate terminal
2. Run: `ollama serve`
3. Then launch voice assistant

### Shortcut Not Working
Make sure you're in the correct directory:
```powershell
cd J:\TOOLS\voice-assistant-rtx\voice-assistant-windows
.\create_shortcut.ps1
```

## What Happens When You Click

1. **VBS file runs** - Shows popup notification
2. **PowerShell launches** - Terminal window opens
3. **Ollama check** - Starts if needed
4. **Conda activates** - `voice-assistant` environment
5. **Kokoro loads** - TTS model initializes (~2 seconds)
6. **Ready!** - Press Ctrl+F2 to talk!

## Hotkeys Reminder

- **Ctrl+F2** - Conversation Mode (AI + Kokoro voice)
- **Ctrl+F1** - Dictation Mode (types what you say)
- **F15** - AI Typing Mode
- **F14** - Screen AI Mode
- **Menu** - Reset conversation
- **Escape** - Exit

---

**Enjoy your one-click voice assistant!** 🎤✨
