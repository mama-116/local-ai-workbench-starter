# 初版実装状況

更新日: 2026-07-20

## 実装済み

- Python 3.12、Fletデスクトップ、httpx、SQLiteによる起動骨格
- UI / Application Service / Domain / Infrastructureの分離
- このPCとプライベートLAN内の複数Ollama接続、逐次応答、停止
- 会話、キャラクター版、モデル設定、発言、分岐、実行状態の永続化
- 発言書き直しと回答再生成で元を残す分岐
- 会話のゴミ箱移動と復元、非active・非root分岐の非表示と復元（物理削除なし）
- 起動時の中断応答復旧
- 解除不能な `strict_free` ガード
- 3カラムのダークUIと独立したメッセージ吹き出し
- 容量を表示するローカルモデル選択と `qwen3.5:9b` の既定推奨
- 明示された言語・形式・内容を優先し、質問へ直接答える既定キャラクター第3版
- 原文併記、非同期処理、再利用、再翻訳、会話単位の自動翻訳ON/OFFに対応した完全ローカル日本語訳（既定は手動）
- 翻訳専用の `llama3.1:latest` と使用モデル表示
- 会話ごとのOllama接続先保存と、公開ネットワークを拒否するLAN限定ガード
- ADR-0008に沿う交換可能なWindows CPU/RAM・NVIDIA GPU/VRAM取得部品
- run単位の観測値、取得元端末、未取得理由のSQLite保存
- ADR-0009に沿う `CONTROL DESK` の直近性能カード（応答時間・生成速度・折りたたみ詳細）
- Fletの起動方法に依存しない `app/.local-data` 固定保存先と復元バックアップ
- UTF-8 `.txt` / `.md` の検証・重複排除・分割に対応したローカルRAG登録Service
- 文書名・文書UUID・チャンクUUID・原文位置を返すSQLiteローカル検索
- 入力欄の添付ボタンによる会話単位のRAG資料登録・選択
- 選択資料だけの会話文脈注入、参照候補表示、AI回答ごとの「資料使用あり／該当箇所なし」と折りたたみ根拠表示
- `CONTROL DESK` ボタンと `Ctrl+Shift+R` による起動確認付きアプリ再起動
- Windows 11 x64向け `Local LLM Chat` ポータブルZIPとSHA-256生成
- 承認済み英数字一時パスでの再現ビルド、入力ファイル限定、完成物のDB・ログ・秘密鍵混入検査
- 配布版のFlet利用者専用データ領域と、配布フォルダーからの会話・設定分離
- Phase 5の許可フォルダー外、リンク経由、秘密情報候補、非対応形式、1MiB超過を拒否する `ReadOnlyToolAccessPolicy`
- Phase 5の許可済みUTF-8テキストを最大20件、相対パス・行番号・短い抜粋付きで返す `BuiltInFolderSearchTool`
- Phase 5の許可済みUTF-8 `.txt` / `.md`を行範囲・64KiB上限付きで返す `BuiltInTextReadTool`
- 内蔵検索・読取りを共通化する `ToolProvider` 契約と、会話単位のフォルダー許可・取消し・SQLite復元
- `pending / running / completed / failed / denied`、入力、PRIVATE結果、件数、サイズ、SHA-256、理由、開始・終了時刻を保存する `tool_calls` 監査
- 1発言3回、逐次実行、1回10秒、結果合計64KiBを強制する `ToolCoordinator`
- Ollamaツール要求の実行と結果返却、最終回答生成、ツール非対応モデルの通常会話フォールバック
- `CONTROL DESK` A案の許可フォルダー、取消し、利用可能ツール、実行中、直近結果、本文非表示の監査履歴導線
- 固定コマンド・引数・SHA-256・Tools-only能力を検査し、会話ごとのMCP Rootsだけを読む同梱 `local-notes MCP`
- SQLite transactionと一意制約で重複claimを防ぎ、失敗・再実行・起動復旧・停止を追記監査するPhase 6 Scheduler
- 総予算5単位・最大5手・最大60秒・固定許可ツールを強制し、PRIVATE外部送信、許可外操作、承認再利用を拒否するPhase 7 Agent実行契約
- 応答確定後に正史記憶候補を有界キューへ渡し、抽出・保存・キュー満杯の失敗を会話成功から分離するApplication契約
- Fake Providerの可逆操作を途中失敗時に逆順復元し、run/step、結果ハッシュ、拒否・復元理由をSQLiteへ追記するAgent監査
- Phase 7Aの固定メモ帳3操作、計画ハッシュ、30秒承認、同一ユーザー・セッション・中整合性、非昇格を検査するComputer Use契約
- 計画全体と一回限り承認を独立再検査し、OS入力を一切送らないFake Desktop Action Broker
- B案の集中レビュー部品。隔離環境未選定、権限不一致、期限切れでは承認ボタンを無効化する
- Computer Useの計画操作数、PRIVATE入力、状態、失敗理由をSQLiteへ追記し、承認をトランザクションで一回だけ消費する監査契約
- CONTROL DESKからB案の集中レビューを開き、OS入力0件のFake Broker実行と再起動後も残るSQLite監査履歴を確認する導線
- 計画の監査作成時から30秒を数える承認カウントダウン。期限切れ時は画面を閉じ、Application層でもFake実行前に拒否して監査へ残す

## 検証済み

- [S-01〜S-09受入記録](acceptance/S01-S09-2026-07-17.md)（独立展開したPhase 0診断と各基準の試験手順・実績）
- `pytest`: 398件成功（Windows固有条件により1件skip）
- `mypy --strict`: 142ファイル、エラー0件
- TurnBatch末尾1か所の再生成操作から専用Application入口へ元AI応答IDとactive branch IDを渡し、成功・競合・二重操作・失敗・キャンセル・会話切替を自動試験で確認。実Ollama `qwen3.5:9b` でも、元TurnBatchを保持した子分岐への再生成とactive branch切替を1回確認
- グループ生成の構造化JSONを受信中に、固定キャストで話者名を再照合した表示専用プレビューをストリーミングし、完了時だけ従来のTurnBatch原子的保存へ進む経路を自動試験で確認
- 5人の `round_table` を実Ollama `qwen3.5:9b` で確認し、登録順の名前付き別吹き出し5件、完了TurnBatch 1件、安定ID付きsegment 5件に加え、入力・出力トークン、Ollama総処理・生成時間、アプリ応答時間、導出速度のSQLite保存と再読込に合格。初回の終了遅延は再現せず、記憶抽出・翻訳・Telemetryの終了後受付停止、待機中翻訳の失敗確定、Telemetryの処理中キャンセル、全資源close続行契約を追加後、実Collector稼働中も0.005秒で終了
- `round_table` 専用Schemaで登録済み正式キャラクターIDまたはナレーター以外を生成段階から除外し、実Ollama `qwen3.5:9b`、4096 token設定の3ターン連続生成が毎回登録順の安定IDで完了することを確認
- Phase 5内蔵Toolsを実Ollamaの `qwen3.5:9b` で画面確認し、異なる2件の検索・読取り・確認コード回答・読取件数、`CONTROL DESK` の `実行中: 1件` から完了への遷移、入力・件数・サイズ・SHA-256・開始終了時刻、監査本文の非表示を確認
- Windows上のFletネイティブ画面起動と3カラム表示
- Ollama Cloud無効化設定を確認し、ローカルモデル15件だけが利用可能になること
- `qwen3.5:9b` の会話応答は初回12.3秒、読込後3.8秒、GPU使用約7.3GiB
- `llama3.1:latest` による外国語原文の非同期日本語訳とSQLite保存
- `192.168.1.17:11434`への読取専用接続とローカルモデル10件の取得
- `CONTROL DESK` の再起動ボタンで新プロセスへ切り替わり、会話データが保持されることを利用者が画面確認
- WindowsポータブルZIP 71.67 MiB、2736エントリ、グループ会話・正史記憶を含むアプリEXEと `LocalNotesMCP.exe` をロック済み本番依存で同梱、禁止ファイル0件、SHA-256一致
- 配布版 `LocalLLMChat.exe` が12秒以上起動を維持し、検査終了後に対象プロセスだけを停止
- 配布版 `LocalNotesMCP.exe` へ実stdio接続し、Tools-only能力、MCP Roots、UTF-8検索・読取り、正常終了を確認
- 別のWindows 11 PCでZIPを展開し、`LocalLLMChat.exe` が起動することを利用者が確認
- 別のWindows 11 PCでOllama会話が動作し、アプリ再起動後も会話履歴が保持されることを利用者が確認
- 未署名EXEの初回起動時にMicrosoft Defender SmartScreenの発行元警告が表示されることを利用者が確認

## 残る配布確認

- 初回Windowsポータブル版の起動、Ollama会話、データ保持、SmartScreen表示まで確認済み。配布に関する未確認項目はなし
- コード署名、カスタムアイコン、インストーラーは、日常利用者が3人以上になるか導入要望が出た時点で再検討

## 初版後へ送るもの

- Phase 7Aの実UI接続、SQLite監査、UI Automation観測、実OS Broker（契約・Fake・B案レビュー部品まで実装済み。隔離環境未選定のため実入力は未実装）
- Phase 7Bの許可フォルダー内の可逆なファイル整理と登録済みPowerShellレシピ（安全境界のみ決定、未実装。Phase 7A受入後に着手）
- 意味検索が必要になった場合のローカル埋め込みモデル比較
- 登録文書が10件を超えた場合の専用文書ライブラリ
- Web検索とプライバシー承認
- [最大5人のグループ会話と出典付き正史記憶](GROUP_CHAT_DESIGN.md)（C案＋非モーダルUndoを選択済み。正史記憶基盤、端末内Ollama抽出器、登録済み明示形の決定論的補完、非同期取得、起動・終了時構成、`TurnBatch` の原子的保存・再起動復元・部分失敗、正式キャスト1〜5人のSQLite台帳と古い生成結果の原子的拒否、1回のOllama構造化生成、通常送信Application Coordinator、名前付き話者別吹き出し、キャストと会話モードの設定UI、正式キャスト全員が知る正史だけを注入する `Memory Context`、記憶の非モーダルUndo・確認待ちUI、グループRunの性能値保存、TurnBatch全体を元分岐から子分岐へ再生成するApplication・SQLite契約・バッチ末尾の再生成UIまで実装済み。`round_table` の完了応答は全キャストが登録順に1回ずつ発言した場合だけ保存し、途中切断は正しい先頭部分だけを許可する。4096 token設定でも日本語をbyte数そのままで過大判定せず、構造化出力を含む上限超過時は最新利用者発言を保持した直近履歴へ縮退する。実Ollama受入で、5人ラウンドテーブルの順序付き別吹き出し、正史記憶UI、性能値の再起動復元、TurnBatch再生成に合格。シーンゲスト、グループ用RAG・Tools、グループ履歴の要約圧縮は未実装）
- 要約・埋め込み用の `asyncio.Queue`
- 回答キャッシュと高度なルーティング
- 観測履歴を使う複数端末・複数モデルの専用比較画面

これらは初版チャットを単独で安定させるため、空の差し込み口や未使用依存を先に追加していません。
