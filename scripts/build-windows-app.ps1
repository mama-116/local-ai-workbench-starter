[CmdletBinding()]
param(
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$appRoot = (Resolve-Path (Join-Path $root 'app')).Path

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $root 'dist'
}

$output = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not $output.StartsWith($root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must stay inside the project: $root"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv was not found. Install uv before building the Windows package.'
}

$pyprojectPath = Join-Path $appRoot 'pyproject.toml'
$pyproject = Get-Content -LiteralPath $pyprojectPath -Raw -Encoding utf8
$versionMatch = [regex]::Match($pyproject, '(?m)^version\s*=\s*"(?<version>\d+\.\d+\.\d+)"\s*$')
if (-not $versionMatch.Success) {
    throw 'A semantic project version was not found in app/pyproject.toml.'
}

$version = $versionMatch.Groups['version'].Value
$packageName = "LocalLLMChat-Windows-x64-$version"
$stageRoot = Join-Path $output $packageName
$zipPath = Join-Path $output ($packageName + '.zip')
$checksumPath = $zipPath + '.sha256'

$safeRemovalRoots = @($stageRoot, $zipPath, $checksumPath)
foreach ($target in $safeRemovalRoots) {
    $resolvedTarget = [System.IO.Path]::GetFullPath($target)
    if (-not $resolvedTarget.StartsWith($root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the project: $resolvedTarget"
    }
}

# MSBuild can corrupt non-ASCII or long workspace paths. The user approved
# this dedicated ASCII-only build directory outside the Japanese workspace.
$approvedBuildRoot = [System.IO.Path]::GetFullPath((Join-Path $env:USERPROFILE 'LocalLLMChatBuild'))
$approvedBuildRootExisted = Test-Path -LiteralPath $approvedBuildRoot
$physicalBuildRoot = Join-Path $approvedBuildRoot 'b'
$physicalBuildRoot = [System.IO.Path]::GetFullPath($physicalBuildRoot)
if (-not $physicalBuildRoot.StartsWith($approvedBuildRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Build directory escaped the approved build root: $physicalBuildRoot"
}
$stagedApp = Join-Path $physicalBuildRoot 'app'
$buildOutput = Join-Path $physicalBuildRoot 'output'
$mcpDist = Join-Path $physicalBuildRoot 'mcp-dist'
$mcpWork = Join-Path $physicalBuildRoot 'mcp-work'
$mcpSpec = Join-Path $physicalBuildRoot 'mcp-spec'
$previousCl = $env:CL

New-Item -ItemType Directory -Force -Path $output | Out-Null
foreach ($target in $safeRemovalRoots) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

$excludedPaths = @(
    '.flet',
    '.local-data',
    '.venv',
    '.mypy_cache',
    '.pytest_cache',
    '__pycache__',
    'build',
    'dist',
    'tests',
    '*.db',
    '*.sqlite*',
    '*.log',
    '.env*'
)

try {
    if (Test-Path -LiteralPath $physicalBuildRoot) {
        Remove-Item -LiteralPath $physicalBuildRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $physicalBuildRoot | Out-Null
    $env:CL = if ([string]::IsNullOrWhiteSpace($previousCl)) { '/utf-8' } else { $previousCl + ' /utf-8' }

    New-Item -ItemType Directory -Force -Path $stagedApp | Out-Null
    $stagedPyproject = Join-Path $stagedApp 'pyproject.toml'
    $stagedRequirements = Join-Path $stagedApp 'requirements.txt'
    Copy-Item -LiteralPath $pyprojectPath -Destination $stagedPyproject -Force
    $stagedConfig = Get-Content -LiteralPath $stagedPyproject -Raw -Encoding utf8
    $dependencyBlockPattern = '(?ms)^dependencies\s*=\s*\[.*?^\]\s*$'
    if ([regex]::Matches($stagedConfig, $dependencyBlockPattern).Count -ne 1) {
        throw 'The staged project dependency block could not be isolated.'
    }
    $stagedConfig = [regex]::Replace(
        $stagedConfig,
        $dependencyBlockPattern,
        'dependencies = []'
    )
    [System.IO.File]::WriteAllText(
        $stagedPyproject,
        $stagedConfig,
        [System.Text.UTF8Encoding]::new($false)
    )
    & uv export --frozen --project $appRoot `
        --no-dev `
        --no-emit-project `
        --no-hashes `
        --format requirements.txt `
        --output-file $stagedRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Locked runtime dependency export failed with exit code $LASTEXITCODE."
    }
    Copy-Item -LiteralPath (Join-Path $appRoot 'README.md') -Destination $stagedApp -Force
    Copy-Item -LiteralPath (Join-Path $appRoot 'src') -Destination $stagedApp -Recurse -Force

    & uv run --frozen --project $appRoot flet build windows $stagedApp `
        --output $buildOutput `
        --project local_llm_chat `
        --artifact LocalLLMChat `
        --product 'Local LLM Chat' `
        --description 'Strictly local, no-charge desktop chat for Ollama' `
        --python-version 3.12 `
        --exclude $excludedPaths `
        --no-rich-output `
        --yes
    if ($LASTEXITCODE -ne 0) {
        throw "flet build windows failed with exit code $LASTEXITCODE."
    }

    $mcpEntry = Join-Path $stagedApp 'src\local_llm_chat\infrastructure\mcp\local_notes_entry.py'
    & uv run --frozen --project $appRoot pyinstaller `
        --noconfirm `
        --clean `
        --onefile `
        --console `
        --name LocalNotesMCP `
        --distpath $mcpDist `
        --workpath $mcpWork `
        --specpath $mcpSpec `
        --paths (Join-Path $stagedApp 'src') `
        $mcpEntry
    if ($LASTEXITCODE -ne 0) {
        throw "LocalNotesMCP build failed with exit code $LASTEXITCODE."
    }

    $executable = Join-Path $buildOutput 'LocalLLMChat.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Windows executable was not created: $executable"
    }

    New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
    Get-ChildItem -LiteralPath $buildOutput -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $stageRoot -Recurse -Force
    }
    $mcpExecutable = Join-Path $mcpDist 'LocalNotesMCP.exe'
    if (-not (Test-Path -LiteralPath $mcpExecutable -PathType Leaf)) {
        throw "Bundled MCP executable was not created: $mcpExecutable"
    }
    Copy-Item -LiteralPath $mcpExecutable -Destination $stageRoot -Force
    Copy-Item -LiteralPath (Join-Path $appRoot 'WINDOWS_PORTABLE_README.txt') -Destination (Join-Path $stageRoot 'はじめに.txt') -Force
    Copy-Item -LiteralPath (Join-Path $root 'THIRD_PARTY_NOTICES.md') -Destination $stageRoot -Force

    $buildInfo = @(
        'Product: Local LLM Chat',
        "Version: $version",
        'Platform: Windows 11 x64',
        'Format: Portable ZIP',
        "BuiltAtUtc: $([DateTime]::UtcNow.ToString('o'))",
        'ModelsIncluded: No',
        'UserDataIncluded: No'
    )
    Set-Content -LiteralPath (Join-Path $stageRoot 'BUILD-INFO.txt') -Value $buildInfo -Encoding utf8

    $portableTestData = Join-Path $physicalBuildRoot 'startup-test-data'
    $portableReadyDirectory = Join-Path $portableTestData 'restart'
    $portableReadyFile = Join-Path $portableReadyDirectory 'portable.ready'
    $portableReadyToken = [guid]::NewGuid().ToString('N')
    New-Item -ItemType Directory -Force -Path $portableReadyDirectory | Out-Null
    $previousPythonPath = $env:PYTHONPATH
    $previousDataDirectory = $env:LOCAL_LLM_CHAT_DATA_DIR
    $previousRestartToken = $env:LOCAL_LLM_CHAT_RESTART_TOKEN
    $previousRestartReadyFile = $env:LOCAL_LLM_CHAT_RESTART_READY_FILE
    $portableTestProcess = $null
    try {
        $env:PYTHONPATH = $null
        $env:LOCAL_LLM_CHAT_DATA_DIR = $portableTestData
        $env:LOCAL_LLM_CHAT_RESTART_TOKEN = $portableReadyToken
        $env:LOCAL_LLM_CHAT_RESTART_READY_FILE = $portableReadyFile
        $portableTestProcess = Start-Process `
            -FilePath (Join-Path $stageRoot 'LocalLLMChat.exe') `
            -WorkingDirectory $stageRoot `
            -WindowStyle Hidden `
            -PassThru
        $portableDeadline = [DateTime]::UtcNow.AddSeconds(20)
        while (
            [DateTime]::UtcNow -lt $portableDeadline `
            -and -not (Test-Path -LiteralPath $portableReadyFile)
        ) {
            Start-Sleep -Milliseconds 250
        }
        if (-not (Test-Path -LiteralPath $portableReadyFile)) {
            throw 'Portable app did not finish initialization within 20 seconds.'
        }
        $reportedToken = Get-Content -LiteralPath $portableReadyFile -Raw -Encoding utf8
        if ($reportedToken -ne $portableReadyToken) {
            throw 'Portable app reported an invalid initialization token.'
        }
    } finally {
        $env:PYTHONPATH = $previousPythonPath
        $env:LOCAL_LLM_CHAT_DATA_DIR = $previousDataDirectory
        $env:LOCAL_LLM_CHAT_RESTART_TOKEN = $previousRestartToken
        $env:LOCAL_LLM_CHAT_RESTART_READY_FILE = $previousRestartReadyFile
        if ($null -ne $portableTestProcess -and -not $portableTestProcess.HasExited) {
            Stop-Process -Id $portableTestProcess.Id -Force
            $portableTestProcess.WaitForExit(5000) | Out-Null
        }
    }

    $forbiddenFilePatterns = @(
        '*.db', '*.sqlite', '*.sqlite3', '*.sqlite3-shm', '*.sqlite3-wal',
        '*.env', '*.key', '*.pfx', '*.log',
        'ollama-connections.json', 'recovery.log', 'server.json'
    )
    $forbiddenDirectoryNames = @('.flet', '.local-data', 'logs', 'backups')

    $forbiddenFiles = Get-ChildItem -LiteralPath $stageRoot -Recurse -Force -File | Where-Object {
        $name = $_.Name
        ($forbiddenFilePatterns | Where-Object { $name -like $_ }).Count -gt 0
    }
    $forbiddenDirectories = Get-ChildItem -LiteralPath $stageRoot -Recurse -Force -Directory | Where-Object {
        $forbiddenDirectoryNames -contains $_.Name
    }
    $privateKeyFiles = Get-ChildItem -LiteralPath $stageRoot -Recurse -Force -File -Filter '*.pem' | Where-Object {
        Select-String -LiteralPath $_.FullName -SimpleMatch -Quiet -Pattern 'PRIVATE KEY-----'
    }

    if ($forbiddenFiles -or $forbiddenDirectories -or $privateKeyFiles) {
        $found = @($forbiddenFiles.FullName) + @($forbiddenDirectories.FullName) + @($privateKeyFiles.FullName)
        throw "Private or runtime data was found in the Windows package. ZIP creation stopped.`n$($found -join [Environment]::NewLine)"
    }

    Compress-Archive -LiteralPath $stageRoot -DestinationPath $zipPath -CompressionLevel Optimal
    $hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
    Set-Content -LiteralPath $checksumPath -Value ($hash.Hash.ToLowerInvariant() + '  ' + [System.IO.Path]::GetFileName($zipPath)) -Encoding ascii

    Write-Output $zipPath
    Write-Output $checksumPath
} finally {
    $env:CL = $previousCl
    if (Test-Path -LiteralPath $physicalBuildRoot) {
        Remove-Item -LiteralPath $physicalBuildRoot -Recurse -Force
    }
    if (-not $approvedBuildRootExisted -and (Test-Path -LiteralPath $approvedBuildRoot)) {
        $remaining = Get-ChildItem -LiteralPath $approvedBuildRoot -Force
        if (@($remaining).Count -eq 0) {
            Remove-Item -LiteralPath $approvedBuildRoot -Force
        }
    }
}
