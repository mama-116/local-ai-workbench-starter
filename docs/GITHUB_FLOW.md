# GitHub Flow 運用ガイド

この文書を、このリポジトリにおけるGitHub Flowの正本とする。
コード、文書、設定、配布物の変更は、すべて短命な作業ブランチとPull Requestを経由して `main` へ反映する。

## 基本原則

- `main` は常に、検証済みで配布可能な状態を保つ。
- `main` へ直接commitまたはpushしない。小さな修正や緊急修正もPull Requestを経由する。
- 1ブランチと1Pull Requestには、原則として1つの目的だけを含める。
- Issueは内部台帳として使う。要否、重複確認、外部投稿承認は [Issue判断と教育的説明の規約](../plugins/local-ai-builder-kit/skills/guide-development/references/issue-and-explanation-policy.md) を正本とし、利用者にIssue番号やGit操作の入力を要求しない。
- PRIVATEデータ、APIキー、トークン、`.env`、DB、個人ログをGitへ追加しない。

## 作業の流れ

1. Issue規約に従って要否を判定し、必要なら既存Issueを検索して、重複がない場合だけ目的と完了条件を持つIssueを作る。
2. 最新の `main` から作業ブランチを作る。
3. 契約、壊れるシナリオのテスト、実装の順で進める。
4. 完了条件単位で、対象ファイルだけをcommitする。
5. テスト、型検査、秘密情報検査、差分レビューを実行する。
6. 作業ブランチをpushし、原則Draft Pull Requestを作る。
7. 検査とレビューが完了したらReadyへ変更し、`main` へmergeする。
8. merge後は作業ブランチを削除し、関連Issueを結果とともに閉じる。

## ブランチ

Codexが作るブランチは `agent/<短い目的>` とする。人が作る場合も、目的が分かる短いkebab-case名を使う。

例:

```text
agent/add-tool-audit-log
agent/document-github-flow
fix/recover-interrupted-run
```

長期間残る統合ブランチ、環境別ブランチ、個人名だけのブランチは作らない。複数フェーズを同時に変更する必要が生じた場合は、先にIssueと依存関係を分割する。

## Commit

- Commitは1つの完了条件または、単独で検証できる中間状態に対応させる。
- メッセージは変更結果を短く表す。Issue番号だけのメッセージにしない。
- 作業ツリーに無関係な変更がある場合、`git add -A` を使わず対象パスだけをstageする。
- 機密情報や生成物を誤ってstageしていないか、commit前に確認する。

## Pull Request

Pull Requestには次を記載する。

- 何を変更したか
- なぜ必要か
- 利用者または開発者への影響
- 実行した検査と結果
- 関連Issue
- 残る未決事項または見直し条件

Draftを既定とし、次を満たすまでReadyにしない。

- 関連テストが成功している
- `mypy --strict` が成功している
- 秘密情報検査で問題がない
- 差分に無関係な変更がない
- 文書、ADR、第三者通知が必要に応じて更新されている

GitHub Actionsによる自動検査が整うまでは、Pull Request本文へローカル検査結果を記録する。自動検査導入後も、画面操作や配布版など自動化できない受入確認は省略しない。

## Mergeと取り消し

通常はSquash mergeを推奨し、Pull Requestの目的を1つの履歴として残す。中間commit自体が障害解析や段階的な検証記録として必要な場合だけ、通常mergeを選ぶ。履歴を直線化するためだけのforce pushや、公開済み `main` の履歴改変は行わない。

merge後に問題が見つかった場合は、`main` を直接修正せず、revertまたは修正用ブランチとPull Requestで戻す。

## 強い反論と見直し条件

この運用への最も強い反論は、小さな文書修正にもPull Requestが必要になり、個人開発では手続きが重くなることである。それでも、公開リポジトリで安全境界と判断履歴を保つ利益を優先する。

Pull Request作成が作業時間の半分以上を継続的に占める、または緊急復旧を妨げた実例が2件以上発生した場合は、軽微変更用テンプレートや自動mergeを検討する。ただし、`main` への直接push禁止と秘密情報検査は維持する。
