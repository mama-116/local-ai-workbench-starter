# ADR-0017: 初回MCPに同梱local-notesサーバーを採用する

- Status: Accepted
- Date: 2026-07-17
- Decision owner: 利用者

## Context

ADR-0014は、内容確認済みの読取専用MCPサーバー1件だけを `stdio` で扱う境界を決めたが、具体的なサーバー、コマンド、許可ツール、配布方法は未決定だった。第三者のFilesystem MCPは書込み機能を同じプロセスへ含み、Playwright MCPはブラウザー操作と外部通信を扱う。LAN内SearXNGは検索語を外部検索サービスへ送るため、いずれもPhase 5の最初の信頼済みプロファイルには範囲が広い。

## Goal and success criteria

アプリと同梱した別プロセスのMCPサーバーから、現在の会話で許可したフォルダー内のUTF-8 `.txt` / `.md`だけを検索・読取りできるようにする。

- アプリが固定した `search_text` と `read_text` だけをモデルへ公開する。
- 起動コマンド、引数、環境変数名、許可ツール名、実行ファイルSHA-256を毎回検証する。
- 許可ルートはMCP Rootsでセッションごとに渡し、モデル入力や固定プロファイルへ絶対パスを含めない。
- サーバー停止、10秒タイムアウト、壊れた応答、未知ツールでも通常チャットを継続し、既存の `tool_calls` へ失敗を記録する。
- MCP Resources、Prompts、Sampling、Tasks、書込み、削除、シェル、公開ネットワークを提供または要求しない。

## Out of scope

- 任意の第三者MCPサーバー登録
- Playwright MCP、SearXNG、Web URL取得
- ファイルの作成、編集、移動、削除
- MCPのHTTP、SSE、Streamable HTTP transport
- 複数MCPサーバーの同時接続や自動再接続

## Decision

同梱する `local-notes MCP` を公式Python SDKの安定版1系で実装する。依存はロックファイルで厳密に解決し、2系を許可しない。開発時は現在のPython実行ファイルを直接起動し、配布時はコンソール用の `LocalNotesMCP.exe` をアプリ本体と同じディレクトリから引数なしで直接起動する。GUIアプリ本体をstdioサーバーとして兼用しない。シェル、パッケージランナー、PATH探索は使わない。

サーバーは `search_text` と `read_text` だけを公開する。MCPクライアント側はサーバー申告を信頼せず、ツール一覧が固定集合と完全一致しない場合は接続を拒否する。各呼出しは会話の許可ルート1件をRootsとして新しいセッションへ渡し、既存の `ReadOnlyToolAccessPolicy` と同じ境界をサーバー側でも適用する。

実行ファイルSHA-256はアプリ起動時に信頼済みプロファイルへ固定し、MCPプロセス起動直前に再計算して一致を要求する。これにより起動後の実行ファイル差替えを拒否する。配布物全体の改ざん検出は既存のZIP SHA-256を正本とする。

## Alternatives

| 案 | 採否 | 理由 |
|---|---|---|
| 同梱local-notes MCP | 採用 | 公開ツール、外部通信、依存、SHA-256をアプリ側で管理できる |
| 公式Filesystem MCP | 不採用 | 同じプロセスが書込み・編集・移動を公開し、Windowsの一般的起動例がシェル経由になる |
| Time MCP | 不採用 | 疎通試験には安全だが、利用者がローカル資料を読む目的を満たさない |
| SearXNG MCP | 延期 | 検索語の外部送信とプライバシー承認をIssue #9で扱う |
| Playwright MCP | 延期 | ブラウザー操作と外部通信をPhase 7以降の承認境界で扱う |
| MCPを実装しない | 不採用 | ToolProvider境界だけではMCP停止・破損応答・stdio lifecycleを検証できない |

## One-way doors

なし。MCP Providerをbootstrapから外せば内蔵Toolsだけへ戻せる。SQLiteスキーマ変更は行わない。

## Risks and rollback

- 子プロセスが終了しない場合は、今回起動したプロセスだけを終了し、ToolCoordinatorへ失敗を返す。
- SDK更新でプロトコル互換性が壊れた場合はロック済み1系へ戻す。
- 配布ランタイムから別プロセスを起動できない場合は、MCP Providerを無効化して内蔵Toolsを継続し、Issueを完了扱いにしない。
- ロールバックはbootstrapからMCP Providerを除外し、依存と同梱サーバーを削除する。既存会話、許可、監査データは変更しない。

## Decision log

| 状態 | 決定 | 理由 | 見直し条件 |
|---|---|---|---|
| 決定 | Issue #4の確認済み1件には同梱local-notes MCPを採用する | 読取専用、stdio、固定SHA、公開ツールを完全に管理できる | 外部MCP固有の機能が必要になった場合 |
| 決定 | SearXNGはIssue #9の検索Providerとして扱う | 検索語の外部送信には独立したプライバシー境界が必要 | MCP経由でなければ実現できない要件が発生した場合 |
| 決定 | Playwright MCPはIssue #4へ含めない | ブラウザー操作と外部通信を伴い、読取専用境界を超える | Phase 7以降または開発用E2Eテストとして評価する場合 |

## Review conditions

- MCP Python SDK 2系が安定し、1系保守終了または重大な脆弱性修正が2系だけへ提供された場合
- 外部MCP固有の機能が必要になり、OS分離、ネットワーク遮断、配布ライセンスを別ADRで決めた場合
- MCP監査履歴が100件以上となり、既存の右側管理欄では追跡しにくくなった場合
