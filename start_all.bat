@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"

echo ============================================
echo  Y式 株価分析ツール 一括起動
echo ============================================
echo.
echo  分析ツール  : http://localhost:8501
echo  バックテスト: http://localhost:8502
echo ============================================
echo.

REM 仮想環境のセットアップ（初回のみ）
if not exist ".venv\Scripts\python.exe" (
    echo 仮想環境を作成中...
    python -m venv .venv
    if errorlevel 1 (
        echo エラー: Python が見つかりません。Python 3.9 以上をインストールしてください。
        pause
        exit /b 1
    )
    echo 依存パッケージをインストール中...
    .venv\Scripts\pip install -q -r requirements.txt
    echo セットアップ完了
    echo.
)

REM 分析ツールを別ウィンドウで起動
echo 分析ツールを起動中（ポート 8501）...
start "株価分析ツール" cmd /k ".venv\Scripts\streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false --theme.base dark --theme.primaryColor #00CC66 --theme.backgroundColor #0E1117 --theme.secondaryBackgroundColor #1E2A35 --theme.textColor #FFFFFF"

REM 少し待機してからバックテストを起動
timeout /t 3 /nobreak > nul

REM バックテストアプリを別ウィンドウで起動
echo バックテストアプリを起動中（ポート 8502）...
start "バックテストアプリ" cmd /k ".venv\Scripts\streamlit run backtest_app.py --server.port 8502 --server.headless true --browser.gatherUsageStats false --theme.base dark --theme.primaryColor #00CC66 --theme.backgroundColor #0E1117 --theme.secondaryBackgroundColor #1E2A35 --theme.textColor #FFFFFF"

REM ブラウザを開く
timeout /t 5 /nobreak > nul
echo ブラウザを開いています...
start http://localhost:8501
timeout /t 2 /nobreak > nul
start http://localhost:8502

echo.
echo 両アプリが起動しました。
echo 終了するには各ウィンドウで Ctrl+C を押してください。
echo.
pause
