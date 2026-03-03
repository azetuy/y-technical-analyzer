@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"

echo ============================================
echo  Y式 株価パターン分析ツール
echo  起動: http://localhost:8501
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

echo ブラウザを自動で開きます...
echo 終了するには Ctrl+C を押してください
echo.

REM 少し待ってからブラウザを開く
start "" cmd /c "timeout /t 4 /nobreak > nul && start http://localhost:8501"

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
