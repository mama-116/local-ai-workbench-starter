# コントリビューションガイド

フィードバックや改善への協力を歓迎します。

## Issueを作成する前に

1. 既存のIssueに同じ内容がないか確認してください。
2. 不具合報告または機能提案のフォームを選んでください。
3. 再現手順や期待する結果を、共有できる範囲で具体的に記載してください。

公開Issueには、APIキー、トークン、`.env` の内容、会話本文、RAG資料、個人ログ、個人を特定できる情報を投稿しないでください。
セキュリティ上の問題に秘密情報が関係する場合は、公開Issueを作成しないでください。

## Pull Request

- 詳細な運用は [GitHub Flow 運用ガイド](docs/GITHUB_FLOW.md) を確認してください。
- `main` へ直接pushせず、作業ブランチからPull Requestを作成してください。
- 変更理由と確認方法を説明してください。
- 関連するテスト、lint、秘密情報検査を実行してください。
- 1つのPull Requestには、できるだけ1つの目的だけを含めてください。

## ローカル検査

Pull Requestを作る前に、リポジトリのルートから次を実行します。依存関係は `app/uv.lock` の内容だけを使い、更新しません。

```powershell
uv run --project app --frozen pytest
uv run --project app --frozen mypy --config-file app/pyproject.toml app/src/local_llm_chat app/tests scripts/check_repository_secrets.py
uv run --project app --frozen python scripts/check_repository_secrets.py --root .
```

失敗した場合は、最初に失敗したコマンドの出力を確認してください。pytestは失敗したテスト名、mypyは `ファイル:行`、秘密情報検査は値を表示せず対象ファイルと理由を示します。秘密情報や個人データが見つかった場合は値をIssueやPRへ貼らず、追跡対象から除外してから再実行してください。

GitHub上では `tests-types-secrets` チェックがPull Requestと `main` へのpushで同じ3検査を実行します。外部サービス障害でない限り、このチェックが成功するまでReadyやmergeにしません。
