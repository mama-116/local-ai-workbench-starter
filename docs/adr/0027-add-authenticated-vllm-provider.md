# ADR-0027: LAN上の認証付きvLLMをProviderとして追加する

- Status: Accepted
- Date: 2026-08-14
- Decision owner: 利用者

## Context

既存アプリはOllamaのHTTP APIだけをProviderとして扱っている。利用者のLANには `192.168.1.17:18080` でvLLMのOpenAI互換サーバーが稼働しており、`/v1/models` は認証なしでは401を返し、APIキー付きでは `deepseek-v4-flash-2bit` を返すことを確認した。vLLMの標準的な接続経路は `/v1/models` と `/v1/chat/completions` である。

会話本文やRAG資料を外部へ送らない、`locality=local` かつ `cost_class=no_charge` だけを許可する、APIキーをGitやログへ持ち込まないという既存の境界は維持する必要がある。一方、現在の `ModelInfo` はOllamaからモデルサイズを取得する前提で、OpenAI互換のモデル一覧だけではサイズを確定できない。

## Decision

既存のProvider契約にvLLM実装を追加し、LANのベースURL `http://192.168.1.17:18080` に対して内部で `/v1/models` と `/v1/chat/completions` を呼び出す。Bearer APIキーは `LOCAL_LLM_CHAT_VLLM_API_KEY` のプロセス環境変数からのみ読み、永続化・表示・監査ログ・エラーメッセージへ含めない。

モデル一覧、モデル検査、SSE逐次応答、停止時のストリーム切断、入力・出力トークン数と経過時間の観測を既存の `LLMProvider` へ投影する。vLLMが返さないモデルサイズは「不明」として扱い、推測値を表示しない。旧形式の接続設定はOllamaとして読み込み、既定のOllama接続と既存会話は変更しない。

## Alternatives

- 何もしない: vLLMへ切り替えられず、利用者のLAN推論基盤を使えないため却下。
- APIキー認証を無効にして接続する: LAN上の他端末が無認証で推論でき、今回確認できた認証境界を弱めるため却下。
- APIキーを接続設定JSONへ保存する: ローカルファイルのコピー・バックアップ・配布検査へ秘密が混入するため却下。
- 既存Ollama ProviderへvLLM分岐を追加する: API形式、エラー、モデルメタデータの差異が一つの実装へ混ざり、撤退とテストの境界が不明瞭になるため却下。

## Consequences

- 接続先選択から既存UIを変えずにvLLMモデルを選べる。
- APIキーが設定されていない場合はvLLM接続だけが利用不可になり、Ollamaへ自動送信先変更はしない。
- vLLMモデルのサイズ表示は「不明」になる。サイズを根拠にした既存のおすすめ判定は、既知サイズのOllamaモデルを優先する。
- vLLMのツール呼出し、JSON Schema出力、チャットテンプレートはモデル・サーバー設定に依存するため、Provider契約試験で保証する範囲を超える機能は個別受入が必要になる。
- APIキーの設定はアプリ起動前の利用者操作として残る。キーを会話へ貼り付けたり、リポジトリへ保存したりしない。

## Review condition

vLLMを複数接続先へ増やす、APIキーをUIで入力・保存する、vLLMのモデルサイズを正確に表示する、またはGroup/Tools/構造化出力の実機互換性を完了条件へ含める場合は、本ADRを見直して別の契約と受入試験を追加する。
