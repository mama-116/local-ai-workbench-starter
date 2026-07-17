# アーキテクチャと安全境界

この文書をデータ境界、承認区分、主要状態の正本とする。

## データフロー

```mermaid
flowchart LR
    U[利用者] --> UI[会話UI]
    UI --> O[会話オーケストレータ]
    O --> F{無料運営ガード}
    F --> P[推論Provider]
    P --> OL[Ollama]
    P -. Phase 7 .-> VL[vLLM on WSL2]
    O --> DB[(SQLiteログ)]
    O --> M[観測Service]
    M --> COL[交換可能な取得部品]
    COL --> DB
    O --> Q[上限付き翻訳Queue]
    Q --> F
    Q --> DB
    F --> R[RAG]
    R --> V[(ローカルVector Store)]
    F --> T[Tool Gateway]
    T --> L[ローカルツール]
    T --> MCP[信頼済みローカルMCP stdio]
    T --> S[SearXNG on LAN]
    T -. 明示設定 .-> D[DuckDuckGo]
    T --> A{承認ゲート}
    A --> X[外部投稿・削除・課金]
    O -. Phase 7 .-> CU[ComputerUseCoordinator]
    CU --> F
    CU --> OBS[画面 + UI Automation観測]
    CU --> CAP{操作Policy + 計画承認}
    CAP --> ACT[固定許可されたローカルUI操作]
    CAP -. Phase 7B .-> FOP[許可フォルダーの可逆ファイル操作]
    CAP -. Phase 7B .-> PSP[登録済みPowerShellレシピ]
    OBS --> DB
    ACT --> DB
    F -. 拒否 .-> CLD[Cloudモデル・有料API]
```

## データ区分

| 区分 | 例 | 外部送信 |
|---|---|---|
| PRIVATE | 会話本文、RAG資料、プロンプト、個人ログ、スクリーンショット、UI構造、入力文 | 禁止 |
| LOCAL_OPERATIONAL | モデル名、速度、CPU/GPU使用率 | 禁止 |
| SEARCH_QUERY | PRIVATEから分離・短縮した検索語 | SearXNGへ送信可 |
| PUBLIC_RESULT | 公開Web検索結果 | ローカル保存可 |

DuckDuckGoを使う場合、検索語は家庭内LANの外へ出る。「完全ローカル」はデータ本体と推論を指し、Web検索語だけを明示的な例外とする。

## 費用境界

初版の費用モードは `strict_free` だけとし、実行中に解除できない。「追加課金ゼロ」は、導入費、ソフト利用料、月額費、API従量課金、クラウドサーバー費が0円であることを指す。既存PC、電気、通常の回線、現在利用中の開発支援環境は含めない。

すべてのProvider、RAG、Tool呼出しは無料運営ガードを通る。各実装は `locality` と `cost_class` を宣言し、初版は `locality=local` かつ `cost_class=no_charge` だけを許可する。値がない、不明、無料枠、試用期間、従量課金、アカウント必須のいずれかであれば拒否する。UIを隠すだけではなく、Application ServiceとProvider境界の両方で検査する。

Ollamaは端末内・LAN内経由でもクラウドモデルを実行できるため、各Ollamaのクラウド無効化確認と、アプリ側のローカルモデル実体確認を併用する。接続先はループバックまたはプライベートIPのHTTPだけを許可し、APIキーとAuthorizationヘッダーを設定・処理・ログへ持ち込まない。

無料運営ガードを通っても、外部検索や外部投稿は別のデータ・承認境界を通る。無料であることは外部送信の許可を意味しない。

## 承認区分

| 区分 | 動作 | 初期値 |
|---|---|---|
| AUTO_READ | ローカル読取、Web検索、状態取得 | 自動 |
| AUTO_WRITE | アプリ内部の許可ワークスペース内の作成・更新 | 自動、全件ログ。Agentによるファイル整理には適用しない |
| AUTO_LOCAL_UI | 利用者が登録した固定アプリ・固定操作・固定上限内のローカルUI操作 | 既定無効、10回連続成功後も利用者が明示有効化 |
| PLAN_APPROVAL_REQUIRED | 起動、クリック、文字入力、Phase 7Bのファイル整理、登録済みPowerShellレシピ | 操作列、対象、入力または展開済みコマンド、予算を一括承認。計画変更で失効 |
| REVERSIBLE_DELETE | ローカルの退避・ゴミ箱移動 | 自動、Undo必須 |
| APPROVAL_REQUIRED | 外部投稿、外部削除、課金、恒久削除 | 毎回承認 |
| DENY | PRIVATEデータの外部送信、許可外パス操作、Cloudモデル、有料・費用不明Provider | 禁止 |

Codex自身は `.codex/config.toml` で `workspace-write` と `on-request` を既定にする。プロジェクトを信頼した場合だけこの設定が読み込まれる。

複数区分に当てはまる場合は、より厳しい区分を優先する。初版のCloudモデルと課金経路は、利用者が承認しても `DENY` のままとする。

## 初期データモデル

- `threads`: 会話の目的と状態
- `messages`: 利用者・AI・ツールの発言
- `message_translations`: 原文UUIDに関連する日本語訳の試行、再利用元、失敗状態
- `runs`: モデル実行単位と設定
- `model_profiles`: Provider、モデル、推論設定
- `prompt_profiles`: system、character、task promptの版
- `tool_calls`: 入力、結果、承認、失敗
- `computer_use_runs`: 画面操作の目的、計画ハッシュ、承認、予算、状態
- `computer_observations`: 前面アプリ、UI構造、スクリーンショットの相対パスとハッシュ
- `computer_actions`: 操作型、対象、引数、リスク、前後観測、状態、失敗
- `artifacts`: 設計書、ADR、コード、モック
- `tasks`: 利用者に見せる次の一手と内部Issue参照
- `documents` / `chunks`: RAG登録資料と断片
- `telemetry_samples`: CPU、RAM、GPU、VRAM、速度
- `scheduled_jobs` / `job_runs`: 定期処理と実行履歴

ログは追記を基本とし、RAGの再構築元になる原文と、検索用派生データを分ける。

`messages.content` は会話履歴へ渡す原文の正本とし、表示用の日本語訳を混ぜない。翻訳は `message_translations` へ試行単位で保存し、状態値は `pending`、`running`、`completed`、`failed` とする。同じ原文ハッシュ、対象言語、Provider、モデルの完了済み結果は別メッセージでも再利用できる。再翻訳は元の試行を上書きせず、新しい試行を追加する。

## Tools/MCP境界

初回のTool Gatewayは、会話ごとに許可された1フォルダーのUTF-8 `.txt` / `.md`検索・読取りと、信頼済みプロファイル1件のローカルMCP `stdio` 接続だけを扱う。書込み、削除、シェル、公開ネットワーク、MCPのResources、Prompts、Sampling、Tasksは扱わない。

`ToolCoordinator` はOllamaが返したツール要求を直接実行せず、現在の会話許可、Provider、ツール名、正規化済みパス、回数・時間・結果サイズ上限を検査する。内蔵ツールとMCPは同じ `ToolProvider` 契約を使う。MCPサーバーの申告は信頼せず、アプリ側の固定プロファイルと許可ツール一覧を正本にする。別プロセスである第三者MCPの挙動はアプリだけでは保証できないため、任意登録は初回対象外とする。

`tool_calls` はUUID、会話、run、許可、Provider、ツール名、入力、状態、結果、時刻、失敗理由、結果ハッシュを保持する。結果本文は上限付きPRIVATEデータとしてSQLiteだけへ保存し、通常ログへ重複出力しない。状態値は `pending`、`running`、`completed`、`failed`、`denied` とし、再実行は別レコードへ追記する。詳細上限、UI、受入基準は [ADR-0014](adr/0014-start-read-only-tools-in-control-desk.md) を正本とする。

画面はApplication層の許可Serviceと監査読取Serviceだけを呼び、ProviderやSQLiteを直接参照しない。監査一覧用の読取結果からはPRIVATEな結果本文を除外し、件数、サイズ、SHA-256、失敗理由、開始・終了時刻だけを返す。

## Computer Use境界

`computer_use_runs` の状態は `pending`、`awaiting_approval`、`running`、`completed`、`failed`、`cancelled`、`denied` とする。`computer_actions` の状態は `proposed`、`approved`、`running`、`completed`、`failed`、`cancelled`、`denied` とする。再実行は別runへ追記し、起動時に残った `running` は `failed` へ復旧して自動再開しない。

初回のComputer Useは、固定されたメモ帳起動プロファイル、UI Automationで検証した要素へのクリック、200文字以内の通常文字入力だけを扱う。モデルは実行ファイル、パス、引数、任意座標、自由なキー列を指定できない。スクリーンショットと画面内文字は信頼できないPRIVATEな観測データであり、利用者の目的、固定許可、承認、予算を変更できない。

操作計画はApplication層の `ComputerUseCoordinator` が `ActionPolicy` へ渡し、対象アプリ、操作型、前面状態、上限、承認を再検査してから1件ずつ実行する。実行直前と直後に画面とUI要素を再取得し、状態が変わった場合は入力せず停止する。UIからOS操作ProviderやSQLiteを直接呼ばない。

スクリーンショット、UI構造、入力文、ウィンドウタイトルはPRIVATEとして利用者データ領域だけへ保存し、通常ログと配布物へ入れない。初回の上限、停止、受入基準は [ADR-0015](adr/0015-start-computer-use-with-approved-notepad-task.md) と [COMPUTER_USE_DESIGN.md](COMPUTER_USE_DESIGN.md) を正本とする。画像座標だけの操作、ゲーム、ブラウザ、外部送信は別ADRまで扱わない。

Phase 7Bでは、`ComputerUseCoordinator` の予算、承認、逐次実行、停止、監査を再利用しつつ、UI操作と分離した `FileOperationProvider` と `PowerShellRecipeProvider` を設ける。Providerはモデルの自由形式出力を実行せず、`ActionPolicy` が検査した固定スキーマの要求1件だけを受け取る。UIは引き続きCoordinatorと監査読取Serviceだけを呼ぶ。

`FileOperationProvider` は会話またはrunへ明示許可したフォルダー配下だけで、作成、コピー、名前変更、移動、ゴミ箱またはアプリ管理退避を扱う。正規化・リンク解決後の入力元と出力先が許可ルート内であることを直前にも再検査し、ジャンクション、シンボリックリンク、reparse point経由の逸脱、既存宛先への上書き、恒久削除を拒否する。変更前メタデータと復元ジャーナルを先に永続化し、復元可能性を検査できない計画は開始しない。

`PowerShellRecipeProvider` はアプリ側で登録したレシピID、型付き引数、固定作業ディレクトリ、時間・出力・process tree上限だけを受け取る。レシピはアプリ同梱で書込み不能な、バージョンとSHA-256を固定した `.ps1` とし、`powershell.exe -File` の引数ベクターからPowerShellのパラメーター束縛で呼び出す。引数をコマンド文字列へ連結せず、`-Command`、`Invoke-Expression`、dot source、実行時のスクリプト生成を禁止する。実行前にレシピ名、型付き引数、対象、影響、上限を表示して計画承認へ束縛し、実行ファイル・引数・作業ディレクトリ・レシピ定義のSHA-256と開始・終了・終了コードを監査する。任意コマンド文字列、`-EncodedCommand`、昇格、公開ネットワーク通信、秘密情報候補へのアクセス、再帰的恒久削除、未登録の実行ファイル・子プロセスを拒否し、停止時は今回生成したprocess treeだけを終了する。

Phase 7Bのファイル整理とPowerShellは常に `PLAN_APPROVAL_REQUIRED` とし、既存の `AUTO_WRITE` や `REVERSIBLE_DELETE` へ自動降格しない。任意PowerShellは `DENY` のままとし、20回連続成功かつ恒久消失、復元失敗、許可外アクセスが0件になった後も、別ADRなしには有効化しない。詳細と受入基準は [ADR-0016](adr/0016-stage-approved-file-and-powershell-operations.md) と [COMPUTER_USE_DESIGN.md](COMPUTER_USE_DESIGN.md) を参照する。

## 観測境界

観測Serviceは会話runの開始・生成中・終了を受け取り、取得部品へ問い合わせる。取得部品はOllama実行情報、WindowsのCPU・RAM、NVIDIAのGPU・VRAM、将来のLAN端末観測を同じ結果形式へ変換する。値には取得元端末と取得時刻を必ず付ける。

観測は補助処理であり、失敗しても会話を停止しない。未取得値を0として扱わず、対応外・タイムアウト・取得失敗を区別する。高頻度データは間引き可能な派生情報として扱い、会話本文やrunの正本を上書きしない。詳細は [ADR-0008](adr/0008-build-observability-as-replaceable-in-app-collectors.md) を正本とする。

初回の観測UIは右側管理欄へ直近runだけを表示する。履歴の表・グラフは同じ保存データを使用する後続画面とし、表示順序と採用理由は [ADR-0009](adr/0009-show-latest-telemetry-in-control-desk-first.md) を正本とする。

無料運営ガードの詳細は [ADR-0004](adr/0004-enforce-strict-free-operation.md) を正本とする。
ローカル日本語訳の表示判断は [ADR-0005](adr/0005-show-local-japanese-translation-inline.md) を正本とする。
