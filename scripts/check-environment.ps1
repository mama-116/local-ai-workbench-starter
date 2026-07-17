[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$results = [System.Collections.Generic.List[object]]::new()
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

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
    if ($memoryGb -ge 31) {
        Add-Result 'Memory' 'PASS' "${memoryGb}GB available (32GB class)"
    } else {
        Add-Result 'Memory' 'WARN' "${memoryGb}GB. 32GB or more is recommended"
    }
} catch {
    Add-Result 'Windows and memory' 'WARN' "System inventory unavailable: $($_.Exception.Message)"
}

foreach ($name in @('git', 'ollama')) {
    $command = Find-Command $name
    if ($command) {
        Add-Result $name 'PASS' $command.Source
    } else {
        Add-Result $name 'WARN' 'Command not found'
    }
}

$uv = Find-Command 'uv'
if ($uv) {
    Add-Result 'uv' 'PASS' $uv.Source
    try {
        $python312 = & $uv.Source python find 3.12 2>$null
        if ($LASTEXITCODE -eq 0 -and $python312) {
            Add-Result 'Python 3.12' 'PASS' ($python312 -join '; ')
        } else {
            Add-Result 'Python 3.12' 'WARN' 'Run uv python install 3.12 before app development'
        }
    } catch {
        Add-Result 'Python 3.12' 'WARN' 'Could not locate Python 3.12 through uv'
    }
} else {
    Add-Result 'uv' 'WARN' 'uv is required for the locked app development environment'
    $python = Find-Command 'python'
    if ($python) {
        $pythonVersion = & $python.Source -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'
        if ($pythonVersion -eq '3.12') {
            Add-Result 'Python 3.12' 'PASS' $python.Source
        } else {
            Add-Result 'Python 3.12' 'WARN' "Python $pythonVersion found; 3.12 is required"
        }
    } else {
        Add-Result 'Python 3.12' 'WARN' 'Python was not found'
    }
}

$codex = Find-Command 'codex'
if ($codex) {
    Add-Result 'Codex' 'PASS' $codex.Source
} else {
    Add-Result 'Codex' 'WARN' 'Continue if this folder can be opened in the Codex app'
}

$manifestPath = Join-Path $root 'plugins\local-ai-builder-kit\.codex-plugin\plugin.json'
$marketplacePath = Join-Path $root '.agents\plugins\marketplace.json'
try {
    $manifest = Get-Content -Raw -Encoding utf8 $manifestPath | ConvertFrom-Json
    $marketplace = Get-Content -Raw -Encoding utf8 $marketplacePath | ConvertFrom-Json
    $entry = $marketplace.plugins | Where-Object { $_.name -eq $manifest.name }
    if (-not $entry) {
        Add-Result 'Local plugin layout' 'FAIL' "Plugin '$($manifest.name)' is missing from the marketplace"
    } elseif ($entry.source.source -ne 'local' -or $entry.source.path -ne './plugins/local-ai-builder-kit') {
        Add-Result 'Local plugin layout' 'FAIL' 'Marketplace source does not point to the bundled plugin'
    } else {
        $invalidSkills = Get-ChildItem -Directory (Join-Path $root 'plugins\local-ai-builder-kit\skills') |
            Where-Object {
                -not (Test-Path -LiteralPath (Join-Path $_.FullName 'SKILL.md')) -or
                -not (Test-Path -LiteralPath (Join-Path $_.FullName 'agents\openai.yaml'))
            }
        if ($invalidSkills) {
            Add-Result 'Local plugin layout' 'FAIL' ('Incomplete skills: ' + (($invalidSkills.Name) -join ', '))
        } else {
            Add-Result 'Local plugin layout' 'PASS' "$($manifest.name); $((Get-ChildItem -Directory (Join-Path $root 'plugins\local-ai-builder-kit\skills')).Count) skills"
        }
    }
} catch {
    Add-Result 'Local plugin layout' 'FAIL' $_.Exception.Message
}

if ($codex) {
    try {
        $pluginList = (& $codex.Source plugin list 2>&1 | Out-String)
        if ($pluginList -match 'local-ai-builder-kit@local-ai-starter\s+installed, enabled') {
            Add-Result 'Local plugin installation' 'PASS' 'local-ai-builder-kit is installed and enabled'
        } else {
            Add-Result 'Local plugin installation' 'WARN' 'Run .\scripts\setup-codex-plugin.ps1, then restart Codex'
        }
    } catch {
        Add-Result 'Local plugin installation' 'WARN' 'Could not read the Codex plugin list'
    }
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
