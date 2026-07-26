[CmdletBinding()]
param(
    [string]$RuntimeDirectory
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$expectedRemote = 'https://github.com/mama-116/local-ai-workbench-starter.git'

if ([string]::IsNullOrWhiteSpace($RuntimeDirectory)) {
    $RuntimeDirectory = Join-Path $repositoryRoot '.local-runtime'
}
$runtimeRoot = [System.IO.Path]::GetFullPath($RuntimeDirectory)
if (-not $runtimeRoot.StartsWith(
    $repositoryRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "最新版用worktreeはプロジェクト内に作成してください: $repositoryRoot"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Gitが見つかりません。'
}

function Invoke-RepositoryGit {
    param(
        [Parameter(Mandatory)]
        [string]$WorkingTree,
        [Parameter(Mandatory)]
        [string[]]$GitArguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $result = & git -C $WorkingTree @GitArguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Gitの実行に失敗しました: git $($GitArguments -join ' ')`n$($result -join [Environment]::NewLine)"
    }
    return @($result | ForEach-Object { $_.ToString() })
}

$remote = @(
    Invoke-RepositoryGit -WorkingTree $repositoryRoot -GitArguments @(
        'remote', 'get-url', 'origin'
    )
)[0].Trim()
if ($remote.TrimEnd('/') -ne $expectedRemote.TrimEnd('/')) {
    throw "想定外のoriginからコードを取得しようとしました: $remote"
}

Invoke-RepositoryGit -WorkingTree $repositoryRoot -GitArguments @(
    'fetch', 'origin', 'main', '--prune'
) | Out-Null

if (Test-Path -LiteralPath $runtimeRoot) {
    if (-not (Test-Path -LiteralPath $runtimeRoot -PathType Container)) {
        throw "最新版用worktreeの場所がフォルダーではありません: $runtimeRoot"
    }
    $repositoryCommonDir = @(
        Invoke-RepositoryGit -WorkingTree $repositoryRoot -GitArguments @(
            'rev-parse', '--path-format=absolute', '--git-common-dir'
        )
    )[0].Trim()
    $runtimeCommonDir = @(
        Invoke-RepositoryGit -WorkingTree $runtimeRoot -GitArguments @(
            'rev-parse', '--path-format=absolute', '--git-common-dir'
        )
    )[0].Trim()
    if ($repositoryCommonDir -ne $runtimeCommonDir) {
        throw "既存フォルダーはこのリポジトリのworktreeではありません: $runtimeRoot"
    }
} else {
    Invoke-RepositoryGit -WorkingTree $repositoryRoot -GitArguments @(
        'worktree', 'add', '--detach', $runtimeRoot, 'origin/main'
    ) | Out-Null
}

$trackedChanges = Invoke-RepositoryGit -WorkingTree $runtimeRoot -GitArguments @(
    'status', '--porcelain', '--untracked-files=no'
)
if ($trackedChanges.Count -gt 0) {
    throw "最新版用worktreeに未保存の変更があります。自動更新せず停止します: $runtimeRoot"
}

Invoke-RepositoryGit -WorkingTree $runtimeRoot -GitArguments @(
    'switch', '--detach', 'origin/main'
) | Out-Null

$entryPoint = Join-Path $runtimeRoot 'Start-LocalLLMChat-Latest.cmd'
if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) {
    throw "最新版ランチャーがmainにありません: $entryPoint"
}

Write-Output $entryPoint
