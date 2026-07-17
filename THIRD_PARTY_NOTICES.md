# Third-party and legacy asset status

この初版ZIPには、第三者または非公開プロジェクトの本文をまだ同梱していない。

| 資産 | 利用予定 | 初版の扱い | 理由 |
|---|---|---|---|
| `fable-perspective` | discussion、spec-authoring、high-level-review | Pluginへ移植し、補助輪Skillから必要時に利用 | 本キット作成者が管理する過去資産として再利用 |
| `ai-dev-kit` | GitHub Flow、開始、検査、出荷フロー | 次版で必要な部分だけ移植 | Proprietaryであり配布範囲の確認が必要 |
| `KAIROS OS` | RAG、Ollama、設計書構成、モック先行 | 設計パターンのみ参照 | Private / All rights reserved |
| cognitive-rhythm-writing Gist | 長文の読みやすさ改善 | URLのみ記録し、本文は未同梱 | ライセンス表記がなく、依存Skillも未同梱 |

## cognitive-rhythm-writing

- Source: https://gist.github.com/k16shikano/eb2929f13ed19c97188393d297be8432
- Upstream dependency: `../japanese-tech-writing/SKILL.md`
- Intended scope: 要件定義書、ADR、比較説明、チュートリアル
- Excluded scope: ボタン、エラー、承認画面など短いUI文言

再配布許可または明確なライセンスを確認できた場合、依存Skillと一緒にPluginへ追加する。

## ローカル実行モデル

モデルはアプリ本体へ同梱せず、利用者のOllama環境に保存する。

| モデル | 用途 | ライセンス | 費用経路 |
|---|---|---|---|
| `qwen3.5:9b` | 会話 | Apache License 2.0 | ローカル実行のみ。API課金なし |
| `llama3.1:latest` | 日本語翻訳 | Llama 3.1 Community License | ローカル実行のみ。API課金なし |

- Qwen3.5 source: https://huggingface.co/Qwen/Qwen3.5-9B
- Llama 3.1 license: `ollama show --license llama3.1:latest` で取得したモデル付属条項
