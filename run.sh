#!/bin/bash
# Y式 株価パターン分析ツール 起動スクリプト

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 仮想環境のセットアップ（初回のみ）
if [ ! -d ".venv" ]; then
    echo "📦 仮想環境を作成中..."
    python3 -m venv .venv
    echo "📥 依存パッケージをインストール中..."
    .venv/bin/pip install -q -r requirements.txt
    echo "✅ セットアップ完了"
fi

echo "🚀 Y式 株価パターン分析ツールを起動中..."
echo "   ブラウザで http://localhost:8501 を開いてください"
echo "   終了するには Ctrl+C を押してください"
echo ""

.venv/bin/streamlit run app.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#00CC66" \
    --theme.backgroundColor "#0E1117" \
    --theme.secondaryBackgroundColor "#1E2A35" \
    --theme.textColor "#FFFFFF"
