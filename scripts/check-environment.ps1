[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$results = [System.Collections.Generic.List[object]]::new()

function Add-Result {
    param(
        [string]$Name,
        [ValidateSet('PASS', 'WARN', 'FAIL')][string]$Status,
        [string]$Detail
    )
    $results.Add([pscustomobject]@{ Check = $Name; Status = $Status; Detail = $Detail })
}

function Find-Command {
    param([string]$Name)
    return Get-Command $Name -ErrorAction SilentlyContinue
}

try {
    $os = Get-CimInstance Win32_OperatingSystem
    if ($os.Caption -match 'Windows 11') {
        Add-Result 'Windows' 'PASS' $os.Caption
    } else {
        Add-Result 'Windows' 'WARN' "Windows 11 is the supported baseline: $($os.Caption)"
    }

    $memoryGb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    if ($memoryGb -ge 32) {
        Add-Result 'Memory' 'PASS' "${memoryGb}GB"
    } else {
        Add-Result 'Memory' 'WARN' "${memoryGb}GB. 32GB or more is recommended"
    }
} catch {
    Add-Result 'Windows and memory' 'WARN' "System inventory unavailable: $($_.Exception.Message)"
}

foreach ($name in @('git', 'python', 'node', 'ollama')) {
    $command = Find-Command $name
    if ($command) {
        Add-Result $name 'PASS' $command.Source
    } else {
        Add-Result $name 'WARN' 'Command not found'
    }
}

$codex = Find-Command 'codex'
if ($codex) {
    Add-Result 'Codex' 'PASS' $codex.Source
} else {
    Add-Result 'Codex' 'WARN' 'Continue if this folder can be opened in the Codex app'
}

$nvidia = Find-Command 'nvidia-smi'
if ($nvidia) {
    $gpu = & $nvidia.Source --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    Add-Result 'NVIDIA GPU' 'PASS' ($gpu -join '; ')
} else {
    Add-Result 'NVIDIA GPU' 'WARN' 'nvidia-smi was not found'
}

$wsl = Find-Command 'wsl'
if ($wsl) {
    Add-Result 'WSL2' 'PASS' 'Available for a later vLLM phase'
} else {
    Add-Result 'WSL2' 'WARN' 'Not required for the Ollama MVP; check again before adding vLLM'
}

try {
    $models = & ollama list 2>$null
    if ($LASTEXITCODE -eq 0) {
        Add-Result 'Ollama connection' 'PASS' (($models | Select-Object -First 3) -join ' / ')
    } else {
        Add-Result 'Ollama connection' 'WARN' 'Ollama was found but did not respond'
    }
} catch {
    Add-Result 'Ollama connection' 'WARN' 'Could not read the Ollama model list'
}

$results | Format-Table -AutoSize -Wrap

if ($results.Status -contains 'FAIL') {
    exit 1
}

Write-Host ''
Write-Host 'Next: open this folder in Codex and send the first request shown in README_FIRST.md.'
