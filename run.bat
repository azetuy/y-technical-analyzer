@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"

echo ============================================
echo  Y式 株価パターン分析ツール
echo  起動: http://localhost:8502
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

echo ブラウザで http://localhost:8502 を開いてください
echo 終了するには Ctrl+C を押してください
echo.

.venv\Scripts\streamlit run app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.base dark ^
    --theme.primaryColor "#00CC66" ^
    --theme.backgroundColor "#0E1117" ^
    --theme.secondaryBackgroundColor "#1E2A35" ^
    --theme.textColor "#FFFFFF"

pause
