Local LLM Chat 0.1.0 - Windowsポータブル版
================================================

このアプリは、Ollamaのローカルモデルだけを使うWindows 11 x64向けチャットです。
Ollamaとモデルは同梱されていません。クラウドAPI、APIキー、月額サービスも使いません。

起動手順
--------
1. ZIPを任意のフォルダーへ展開します。
2. Ollamaを起動します。
3. LocalLLMChat.exeをダブルクリックします。

初回準備
--------
- Ollamaを別途インストールし、使用するモデルをOllamaへ保存してください。
- このPCと接続するLAN端末の次のファイルで、Ollama Cloudを無効にしてください。

  %USERPROFILE%\.ollama\server.json

  {
    "disable_ollama_cloud": true
  }

- 設定変更後はOllamaを再起動してください。
- LAN端末はアプリの「接続先を追加・編集」から登録できます。

データの保存
------------
会話、設定、接続先、実行ログは、Windowsの利用者ごとに分離されたFletのアプリデータ領域へ保存されます。
LocalLLMChat.exeの隣には保存されないため、配布フォルダーを置き換えても会話は残ります。

安全と費用
----------
- アプリはlocalhostまたはプライベートLAN内のOllamaだけへ接続します。
- Cloudモデル、有料API、費用が不明な接続先への送信は拒否します。
- 会話本文、RAG資料、個人アーカイブを外部APIへ送信しません。
- この配布物はコード署名していないため、初回起動時にMicrosoft Defender SmartScreenの発行元警告が表示されます。
- 先に下記のSHA-256を照合し、配布元と値が一致した場合だけ「詳細情報」から `LocalLLMChat.exe` を確認して「実行」を選んでください。
- ファイル名またはSHA-256が配布元の案内と異なる場合は実行しないでください。

配布物の確認
------------
ZIPと一緒にある .sha256 ファイルの値を、PowerShellの次の結果と比較できます。

  Get-FileHash .\LocalLLMChat-Windows-x64-0.1.0.zip -Algorithm SHA256

問題が起きた場合
----------------
- アプリが開かない: Windows 11 x64であることを確認してください。
- モデルが表示されない: Ollamaが起動中か、「状態を再確認」で確認してください。
- LAN端末へつながらない: 相手側Ollamaの待受設定とWindows Firewallを確認してください。
- 無料運営が送信停止になる: 各Ollama端末のserver.jsonとCloud無効化確認を見直してください。
