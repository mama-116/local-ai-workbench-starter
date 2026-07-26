# Local LLM Chat

Ollamaのローカルモデルだけを使う、Windows向けFletデスクトップチャットです。外部ブラウザ、クラウドAPI、APIキー、月額サービスは使いません。

## 初版でできること

- 1つの会話につき1人のキャラクターを選んだストリーミング会話
- 複数キャラクターの作成と、変更履歴を残す版管理
- 最大5人の正式キャストによるグループ会話と、話者別のストリーミング表示
- グループ会話のTurnBatch単位の保存・分岐再生成と、部分応答の安全な保持
- 出典付き正史記憶の自動保存・Undoと、機微情報の承認・却下
- 会話ごとのローカルモデル切替
- このPCとLAN内端末にある複数Ollama接続先の切替
- 利用者発言の書き直しとAI回答の再生成（元の続きは分岐として保持）
- 分岐の往復、会話の保管と復元
- SQLiteへの会話・設定・途中応答・実行状態の保存
- `.txt` / `.md` 資料を使うローカルRAGと、回答に使った根拠の表示
- 外国語回答の原文を残した完全ローカル日本語訳
- 直近回答の応答時間・生成速度・CPU・RAM・GPU・VRAM表示
- アプリ内ボタンと `Ctrl+Shift+R` による起動確認付き再起動
- Ollama切断、モデル不在、DB障害、無料条件違反の画面通知

Web検索、回答キャッシュ、意味ベクトル検索、シーンゲスト、グループ会話専用RAG・Tools、トークン単位の文脈圧縮は初版の対象外です。

## 無料運営の前提

初期接続先はこのPCの `http://127.0.0.1:11434` と、LAN端末の `http://192.168.1.17:11434` です。公開IP、ホスト名、HTTPS、認証情報付きURLは登録できません。

このPCでは、Ollamaがlocalhost経由でCloudモデルを使わないよう、次のファイルを利用者が設定してOllamaを再起動する必要があります。

`%USERPROFILE%\.ollama\server.json`

```json
{
  "disable_ollama_cloud": true
}
```

アプリはこのファイルを自動変更しません。LAN端末でも同じCloud無効化を行い、接続先の編集画面で確認済みにする必要があります。設定未確認、Cloudモデル名、ローカル実体を確認できないモデルのいずれかでは送信を停止します。接続先設定は `ollama-connections.json`、会話ごとの接続先はSQLiteへ保存されます。

## 開発版の起動

リポジトリのルートで実行します。

```powershell
uv sync --project app
uv run --project app flet run app\src\main.py
```

検証コマンド:

```powershell
uv run --project app pytest app\tests
uv run --project app mypy --config-file app\pyproject.toml app\src\local_llm_chat app\src\main.py app\tests
```

## Windowsポータブル版の作成

Windows 11 x64で、リポジトリのルートから実行します。Visual Studioの「C++によるデスクトップ開発」とWindowsの開発者モードが必要です。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-app.ps1
```

`dist/LocalLLMChat-Windows-x64-0.1.0.zip` とSHA-256ファイルを生成します。ビルド中だけ、承認済みの英数字パス `%USERPROFILE%\LocalLLMChatBuild\b` を使用し、終了時に削除します。配布物は未署名のため、Windowsの発行元確認が表示される場合があります。

配布版ではFletの `FLET_APP_STORAGE_DATA`、開発用の直接起動では `app/.local-data` にSQLiteを保存します。物理削除機能はありません。

## この開発PCでmerge済み最新版を起動

普段の作業ブランチを変更せず、GitHubへmerge済みの `main` だけを取得・ビルドして起動する場合は、リポジトリのルートから初回に次を実行します。

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\install-latest-launcher.ps1
```

表示された `.local-runtime\Start-LocalLLMChat-Latest.cmd` を以後の起動入口にします。起動時に `origin/main` を確認し、新しいcommitだけをビルドします。取得またはビルドに失敗した場合は、成功確認済みの直前版を起動します。普段の作業worktreeと未コミット変更には触れません。
