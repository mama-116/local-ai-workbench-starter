[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$marketplaceFile = Join-Path $root '.agents\plugins\marketplace.json'
$manifestFile = Join-Path $root 'plugins\local-ai-builder-kit\.codex-plugin\plugin.json'

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw 'Codex CLI was not found. Open this folder in the Codex app and try again.'
}

if (-not (Test-Path -LiteralPath $marketplaceFile)) {
    throw "Marketplace file was not found: $marketplaceFile"
}
if (-not (Test-Path -LiteralPath $manifestFile)) {
    throw "Plugin manifest was not found: $manifestFile"
}

$marketplace = Get-Content -Raw -Encoding utf8 $marketplaceFile | ConvertFrom-Json
$manifest = Get-Content -Raw -Encoding utf8 $manifestFile | ConvertFrom-Json
$entry = $marketplace.plugins | Where-Object { $_.name -eq $manifest.name }

if (-not $entry) {
    throw "Plugin '$($manifest.name)' is not registered in $marketplaceFile"
}
if ($entry.source.source -ne 'local' -or $entry.source.path -ne './plugins/local-ai-builder-kit') {
    throw 'The marketplace source does not point to the bundled local plugin.'
}

$configured = (& codex plugin marketplace list --json | ConvertFrom-Json).marketplaces |
    Where-Object { $_.name -eq $marketplace.name }

if ($configured) {
    $configuredRoot = [System.IO.Path]::GetFullPath($configured.root)
    if ($configuredRoot -ne $root) {
        throw "Marketplace '$($marketplace.name)' already points to another folder: $configuredRoot"
    }
} else {
    & codex plugin marketplace add $root --json
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not register the local marketplace.'
    }
}

& codex plugin add "$($manifest.name)@$($marketplace.name)" --json
if ($LASTEXITCODE -ne 0) {
    throw 'Could not install the local plugin.'
}

Write-Host "Installed $($manifest.name) from $($marketplace.name)."
Write-Host 'Restart Codex and open this folder in a new task.'
