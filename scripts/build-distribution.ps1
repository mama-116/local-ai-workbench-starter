[CmdletBinding()]
param(
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $root 'dist'
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)

if (-not $output.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must stay inside the project: $root"
}

$forbiddenPatterns = @(
    '*.env', '*.pem', '*.key', '*.pfx', '*.db', '*.sqlite', '*.sqlite3',
    'archive', 'raw', 'checksums', 'backups', 'logs', 'data'
)

$forbidden = Get-ChildItem -LiteralPath $root -Recurse -Force -File | Where-Object {
    $item = $_
    $forbiddenPatterns | Where-Object {
        $item.Name -like $_ -or $item.DirectoryName.Split([System.IO.Path]::DirectorySeparatorChar) -contains $_
    }
}

if ($forbidden) {
    $list = ($forbidden.FullName -join [Environment]::NewLine)
    throw "Forbidden distribution files were found. ZIP creation stopped.`n$list"
}

$allowed = @(
    'README_FIRST.md', 'AGENTS.md', '.gitignore', 'THIRD_PARTY_NOTICES.md',
    '.agents', '.codex', 'docs', 'plugins', 'scripts'
)

$stage = Join-Path $env:TEMP ('local-ai-workbench-starter-' + [guid]::NewGuid().ToString('N'))
$packageRoot = Join-Path $stage 'local-ai-workbench-starter'
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

try {
    foreach ($relative in $allowed) {
        $source = Join-Path $root $relative
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $packageRoot -Recurse -Force
        }
    }

    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $zip = Join-Path $output 'local-ai-workbench-starter-phase0.zip'
    if (Test-Path -LiteralPath $zip) {
        Remove-Item -LiteralPath $zip -Force
    }
    Compress-Archive -LiteralPath $packageRoot -DestinationPath $zip -CompressionLevel Optimal
    Write-Output $zip
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}
