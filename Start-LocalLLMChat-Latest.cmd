@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0scripts\start-latest-main.ps1"
if errorlevel 1 (
  echo.
  echo 最新版を起動できませんでした。上のメッセージを確認してください。
  pause
)
