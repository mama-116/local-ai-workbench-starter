# ADR-0018: 追記型台帳を持つinterval Schedulerを採用する

- Status: Accepted
- Date: 2026-07-17
- Decision owner: 配布者

## Context

Phase 6は、定期処理の重複防止、失敗、再実行、再起動復旧、強制停止を単独で検証する段階である。具体的な定期タスクと操作UIはまだ決まっていない。OSタスクスケジューラを使うとWindows固有の登録・権限・削除が先に必要になり、Phase 6のApplication契約とSQLite監査を単独検証しにくい。

## Goal and success criteria

- 同一ジョブ・同一予定時刻の初回実行をSQLite制約で1件にする。
- 失敗と再実行を別の `job_runs` 行へ追記する。
- 起動時に残った `pending` / `running` を失敗へ復旧し、自動再開しない。
- 停止時はこのSchedulerが開始した処理だけをcancelし、失敗理由を保存する。
- 具体的なタスク、UI、外部通信、書込み権限をPhase 6へ混ぜない。

## Decision

Application層に、固定間隔の予定を扱うin-process Schedulerを置き、bootstrapから1秒間隔のdriverを開始して終了時に停止する。`scheduled_jobs` はジョブ定義、`job_runs` は試行台帳とし、初回試行は `(job_id, scheduled_for, attempt=1)`、手動再実行は同じ予定時刻へattemptを増やして追記する。同一ジョブで `running` は最大1件とする。

予定の取得と `running` 行の作成は1つのSQLite書込みトランザクションで行う。プロセス内ロックだけには依存せず、複数Schedulerインスタンスから同時取得しても一方だけが成功する。取りこぼした間隔は1 tickあたり最大10件まで古い順に処理し、無制限catch-upを避ける。

ハンドラーはアプリ側で名前と実装を固定登録する。DBのpayloadから任意コード、シェル、外部Providerを選べない。Issue #6では具体的ハンドラーとUIを追加せず、後続Issueが目的と安全境界を決めて登録する。

## Alternatives

| 案 | 採否 | 理由 |
|---|---|---|
| in-process + SQLite台帳 | 採用 | 既存境界内で重複、復旧、再実行、停止を決定論的に検証できる |
| Windows Task Scheduler | 不採用 | OS登録と権限変更が先行し、ポータブル配布と単体試験を複雑にする |
| APScheduler等の追加依存 | 不採用 | 初回要件は固定間隔と追記台帳で足り、依存追加の利益が小さい |
| UIと実タスクを同時実装 | 延期 | 利用者の用途・通知方法・権限を勝手に決めることになる |

## Risks and rollback

- 長時間停止後のcatch-upが負荷になるため、1 tick上限を10件とする。
- ハンドラーがcancelを無視する場合、Application層だけではプロセス強制終了を保証できない。後続の外部プロセス型タスクはProvider側でprocess tree停止を追加する。
- ロールバックはbootstrapからSchedulerを外す。追記済み台帳は監査として保持する。

## Decision log

| 状態 | 決定 | 理由 | 見直し条件 |
|---|---|---|---|
| 決定 | Phase 6は汎用Scheduler基盤だけを完成させる | Issueの単独完了条件を検証し、未選択のタスクやUIを混ぜない | 最初の定期タスクを選ぶIssueへ着手するとき |
| 決定 | SQLite制約と単一transactionで重複取得を防ぐ | process内ロックだけでは再起動・複数インスタンスを防げない | 複数端末で同じDBを共有する要件が発生したとき |
| 決定 | 再実行は元行を更新せずattemptを増やす | 失敗した事実と再試行結果を両方追跡する | 保存件数が運用を圧迫したとき |

## Review conditions

- calendar/cron、timezone、missed-run coalescingが必要になった場合
- 実ジョブが外部プロセスを起動し、cancelだけでは停止できない場合
- 監査件数が1万件を超え、保持期間が必要になった場合
