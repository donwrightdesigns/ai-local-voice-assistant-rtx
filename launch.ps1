#!/usr/bin/env pwsh
<#
.SYNOPSIS
    RTX Hands-Free Assistant - Quick Launch Wrapper
    
.DESCRIPTION
    Convenience wrapper that delegates to scripts\launch.ps1
    Run from repo root for easy access.
    
.EXAMPLE
    .\launch.ps1
    .\launch.ps1 --device cpu
    .\launch.ps1 --profile advanced
#>

param(
    [string]$device = "",
    [string]$profile = "",
    [string]$keywords = "",
    [string]$provider = ""
)

$scriptPath = Join-Path $PSScriptRoot "scripts\launch.ps1"

if (-not (Test-Path $scriptPath)) {
    Write-Host "❌ Launch script not found at: $scriptPath" -ForegroundColor Red
    exit 1
}

# Forward all args to the actual launcher
& $scriptPath @args
