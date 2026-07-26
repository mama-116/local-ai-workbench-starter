[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
$runtimeRoot = [System.IO.Path]::GetFullPath($RepositoryRoot)
$expectedRemote = 'https://github.com/mama-116/local-ai-workbench-starter.git'
$buildRoot = Join-Path $runtimeRoot 'dist\latest-main'
$readyMarkerName = 'BUILD-READY.json'
$mutex = [System.Threading.Mutex]::new(
    $false,
    'Local\LocalLLMChatLatestMainLauncher'
)
$lockAcquired = $false

function Invoke-RuntimeGit {
    param(
        [Parameter(Mandatory)]
        [string[]]$GitArguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $result = & git -C $runtimeRoot @GitArguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Gitの実行に失敗しました: git $($GitArguments -join ' ')`n$($result -join [Environment]::NewLine)"
    }
    return @($result | ForEach-Object { $_.ToString() })
}

function Read-ReadyBuild {
    param(
        [Parameter(Mandatory)]
        [string]$MarkerPath
    )

    try {
        $marker = Get-Content -LiteralPath $MarkerPath -Raw -Encoding utf8 |
            ConvertFrom-Json
        if ($marker.commit -notmatch '^[0-9a-f]{40}$') {
            return $null
        }
        $commitDirectory = [System.IO.Path]::GetFullPath(
            (Split-Path -Parent $MarkerPath)
        )
        if ((Split-Path -Leaf $commitDirectory) -ne $marker.commit) {
            return $null
        }
        $executable = [System.IO.Path]::GetFullPath(
            (Join-Path $commitDirectory ([string]$marker.executable))
        )
        if (-not $executable.StartsWith(
            $commitDirectory + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            return $null
        }
        if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
            return $null
        }
        return [pscustomobject]@{
            Commit = [string]$marker.commit
            Executable = $executable
            MarkerPath = $MarkerPath
        }
    } catch {
        return $null
    }
}

function Find-LatestReadyBuild {
    if (-not (Test-Path -LiteralPath $buildRoot -PathType Container)) {
        return $null
    }
    $markers = Get-ChildItem -LiteralPath $buildRoot -Filter $readyMarkerName `
        -Recurse -File | Sort-Object LastWriteTimeUtc -Descending
    foreach ($markerFile in $markers) {
        $readyBuild = Read-ReadyBuild -MarkerPath $markerFile.FullName
        if ($null -ne $readyBuild) {
            return $readyBuild
        }
    }
    return $null
}

try {
    try {
        $lockAcquired = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $lockAcquired = $true
    }
    if (-not $lockAcquired) {
        throw '最新版の確認またはビルドがすでに実行中です。'
    }

    if (-not $NoLaunch) {
        $runningApp = Get-Process -Name 'LocalLLMChat' -ErrorAction SilentlyContinue
        if ($null -ne $runningApp) {
            Write-Output 'Local LLM Chatはすでに起動しています。'
            return
        }
    }

    $selectedBuild = $null
    try {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            throw 'Gitが見つかりません。'
        }
        $remote = @(
            Invoke-RuntimeGit -GitArguments @('remote', 'get-url', 'origin')
        )[0].Trim()
        if ($remote.TrimEnd('/') -ne $expectedRemote.TrimEnd('/')) {
            throw "想定外のoriginからコードを取得しようとしました: $remote"
        }

        Invoke-RuntimeGit -GitArguments @(
            'fetch', 'origin', 'main', '--prune'
        ) | Out-Null
        $targetCommit = @(
            Invoke-RuntimeGit -GitArguments @(
                'rev-parse', 'refs/remotes/origin/main^{commit}'
            )
        )[0].Trim().ToLowerInvariant()
        if ($targetCommit -notmatch '^[0-9a-f]{40}$') {
            throw "mainのcommitを確認できませんでした: $targetCommit"
        }

        $trackedChanges = Invoke-RuntimeGit -GitArguments @(
            'status', '--porcelain', '--untracked-files=no'
        )
        if ($trackedChanges.Count -gt 0) {
            throw '最新版用worktreeに未保存の変更があるため、自動更新しません。'
        }

        $currentCommit = @(
            Invoke-RuntimeGit -GitArguments @('rev-parse', 'HEAD')
        )[0].Trim().ToLowerInvariant()
        if ($currentCommit -ne $targetCommit) {
            Invoke-RuntimeGit -GitArguments @(
                'switch', '--detach', $targetCommit
            ) | Out-Null
        }

        $commitBuildRoot = Join-Path $buildRoot $targetCommit
        $readyMarker = Join-Path $commitBuildRoot $readyMarkerName
        $selectedBuild = Read-ReadyBuild -MarkerPath $readyMarker
        if ($null -eq $selectedBuild) {
            if (Test-Path -LiteralPath $readyMarker -PathType Leaf) {
                Remove-Item -LiteralPath $readyMarker -Force
            }
            $buildScript = Join-Path $runtimeRoot 'scripts\build-windows-app.ps1'
            if (-not (Test-Path -LiteralPath $buildScript -PathType Leaf)) {
                throw "Windowsビルドスクリプトが見つかりません: $buildScript"
            }
            & powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned `
                -File $buildScript `
                -OutputDirectory $commitBuildRoot
            if ($LASTEXITCODE -ne 0) {
                throw "最新版のビルドに失敗しました。終了コード: $LASTEXITCODE"
            }

            $executables = @(
                Get-ChildItem -LiteralPath $commitBuildRoot `
                    -Filter 'LocalLLMChat.exe' -Recurse -File
            )
            if ($executables.Count -ne 1) {
                throw "起動対象のEXEを1件に特定できませんでした: $commitBuildRoot"
            }
            $commitBuildPrefix = (
                [System.IO.Path]::GetFullPath($commitBuildRoot).TrimEnd(
                    [System.IO.Path]::DirectorySeparatorChar
                ) + [System.IO.Path]::DirectorySeparatorChar
            )
            $executablePath = [System.IO.Path]::GetFullPath(
                $executables[0].FullName
            )
            if (-not $executablePath.StartsWith(
                $commitBuildPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                throw "起動対象のEXEがcommit別ビルド領域の外にあります: $executablePath"
            }
            $relativeExecutable = $executablePath.Substring(
                $commitBuildPrefix.Length
            )
            $marker = [ordered]@{
                commit = $targetCommit
                executable = $relativeExecutable
                built_at_utc = [DateTime]::UtcNow.ToString('o')
            }
            New-Item -ItemType Directory -Force -Path $commitBuildRoot |
                Out-Null
            $marker | ConvertTo-Json |
                Set-Content -LiteralPath $readyMarker -Encoding utf8
            $selectedBuild = Read-ReadyBuild -MarkerPath $readyMarker
            if ($null -eq $selectedBuild) {
                throw '成功済みビルドの記録を検証できませんでした。'
            }
        }
    } catch {
        Write-Warning "最新版を準備できませんでした。直前の成功版を確認します。 $($_.Exception.Message)"
        $selectedBuild = Find-LatestReadyBuild
    }

    if ($null -eq $selectedBuild) {
        throw '起動できる成功済みビルドがありません。ネットワークとビルド環境を確認してください。'
    }

    Write-Output "起動commit: $($selectedBuild.Commit)"
    Write-Output "起動EXE: $($selectedBuild.Executable)"
    if (-not $NoLaunch) {
        Start-Process -FilePath $selectedBuild.Executable `
            -WorkingDirectory (Split-Path -Parent $selectedBuild.Executable)
    }
} finally {
    if ($lockAcquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
