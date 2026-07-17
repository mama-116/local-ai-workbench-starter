# Local AI Workbench Starter

このフォルダーは、Codexとの会話を中心にLocalLLMアプリを段階的に作るための配布キットです。

## 最初の1回だけ

1. ZIPを展開して、このフォルダーをCodexで開く。
2. PowerShellで `powershell -ExecutionPolicy Bypass -File .\scripts\setup-codex-plugin.ps1` を1回だけ実行する。
3. Codexを再起動する。
4. 新しいタスクでこのフォルダーを開く。

このスクリプトは、同梱した `Local AI Starter` marketplaceを登録し、`local-ai-builder-kit` をインストールする。失敗した場合でも、`AGENTS.md`による基本案内は動作するため、表示されたエラーをCodexへ相談する。

セットアップ状態を再確認する場合は、`powershell -ExecutionPolicy Bypass -File .\scripts\check-environment.ps1` を実行する。

準備後、次の一文だけを送ってください。

> 作りたいものを相談したいです。専門用語を減らし、質問は一度に3つまでにしてください。

Codexは、いきなり実装を始めず、次の順番で案内します。

1. 作りたいものを聞く
2. やらないことを決める
3. 複数案を理由付きで比較する
4. 画面はモックを見て選ぶ
5. 決定をADRと設計書へ残す
6. 30分以内の最初の作業を1つだけ提示する
7. テストとレビューを通してから次へ進む

Git、Issue、ブランチ、ADRは必要に応じてCodexが扱います。利用者が仕組みを覚えることは完了条件ではありません。

## このキットが守る境界

- 会話、RAG資料、プロンプト、実行ログはローカル保存を既定とする
- Web検索へ会話全文やRAG資料を送らない
- 外部への投稿、外部データの削除、課金を伴う操作は毎回承認を求める
- ローカルの恒久削除は避け、原則として復元可能な退避を使う
- 個人アーカイブや過去プロジェクトを配布物へ直接含めない

詳細は `docs/REQUIREMENTS.md` を参照してください。
