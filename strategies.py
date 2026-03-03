"""
バックテスト戦略定義

各戦略は backtester.BacktestEngine.run() に渡す dict として定義する。

キー一覧:
  entry_condition   : "kaitetsu" | "rsi_oversold" | "macd_golden" |
                      "composite" | "kaitetsu_plus_composite" | "bb_squeeze_break"
  entry_threshold   : エントリー閾値（戦略依存）
  exit_stop         : "ma5_break" | "macd_dead" | "composite_negative" | "rsi_overbought"
  stop_type         : "ma5" | "fixed_pct" | "atr"
  stop_pct          : fixed_pct の場合の損切り幅 (%)
  target_type       : "resistance" | "rr_ratio" | "none"
  target_rr         : rr_ratio の場合のリスクリワード比
  max_holding_days  : タイムストップ（営業日）
  position_size_pct : ポジションサイズ（資金の %）
  description       : 戦略の説明
  reference         : 参考文献・根拠
"""

STRATEGIES: dict[str, dict] = {

    # ──────────────────────────────────────────
    # 1. Y式買鉄（完全版）
    # ──────────────────────────────────────────
    "Y式買鉄（完全版）": {
        "entry_condition": "kaitetsu",
        "entry_threshold": 5,          # 5条件すべて満たす
        "exit_stop": "ma5_break",      # 5MA割れで手仕舞い
        "stop_type": "ma5",            # ストップは5MA水準
        "target_type": "resistance",   # 直近抵抗線を目標
        "max_holding_days": 30,
        "position_size_pct": 10.0,
        "description": (
            "Y式テクニカル分析の核心。"
            "5MA > 20MA・20MA上向き・5MA転換・ボディ50%以上・MA20接近の"
            "5条件すべてを満たした時のみエントリー。"
            "トレンドフォロー系で最も厳格なフィルタリング。"
        ),
        "reference": "Yさん講義 完全版（Obsidianノート）",
        "emoji": "⭐",
    },

    # ──────────────────────────────────────────
    # 2. Y式買鉄（緩和版）
    # ──────────────────────────────────────────
    "Y式買鉄（4条件）": {
        "entry_condition": "kaitetsu",
        "entry_threshold": 4,          # 5条件中4つ以上
        "exit_stop": "ma5_break",
        "stop_type": "ma5",
        "target_type": "resistance",
        "max_holding_days": 25,
        "position_size_pct": 8.0,
        "description": (
            "買鉄5条件のうち4条件を満たした場合にエントリー。"
            "トレード機会を増やしつつ基本的なフィルタは維持。"
            "完全版と比較してどちらが実戦的かを検証する目的。"
        ),
        "reference": "Yさん講義 完全版（Obsidianノート）",
        "emoji": "⭐",
    },

    # ──────────────────────────────────────────
    # 3. RSI逆張り戦略
    # ──────────────────────────────────────────
    "RSI逆張り": {
        "entry_condition": "rsi_oversold",
        "entry_threshold": 30,         # RSI 30以下から反転
        "exit_stop": "rsi_overbought",
        "exit_target_threshold": 65,   # RSI 65以上で利確
        "stop_type": "atr",            # ATR×2 のストップ
        "target_type": "rr_ratio",
        "target_rr": 2.5,
        "max_holding_days": 20,
        "position_size_pct": 8.0,
        "description": (
            "RSI(14) が売られすぎゾーン（30以下）から回復した際にエントリー。"
            "ただし 20MA が上向きの場合のみ（トレンドに逆らわない）。"
            "Wilder(1978) の原著に基づくクラシックな逆張り戦略。"
        ),
        "reference": "Wilder, J.W. (1978). New Concepts in Technical Trading Systems.",
        "emoji": "📊",
    },

    # ──────────────────────────────────────────
    # 4. MACDゴールデンクロス戦略
    # ──────────────────────────────────────────
    "MACDクロス": {
        "entry_condition": "macd_golden",
        "exit_stop": "macd_dead",      # デッドクロスで手仕舞い
        "stop_type": "fixed_pct",
        "stop_pct": 6.0,               # 6%固定損切り
        "target_type": "resistance",
        "max_holding_days": 40,
        "position_size_pct": 8.0,
        "description": (
            "MACD(12,26,9) のゴールデンクロスでエントリー、"
            "デッドクロスで手仕舞い。"
            "Gerald Appel が開発したトレンドフォロー指標の基本活用法。"
            "比較的長めの保有期間に適している。"
        ),
        "reference": "Appel, G. (2005). Technical Analysis: Power Tools for Active Investors.",
        "emoji": "📈",
    },

    # ──────────────────────────────────────────
    # 5. 複合スコア戦略
    # ──────────────────────────────────────────
    "複合スコア戦略": {
        "entry_condition": "composite",
        "entry_threshold": 3.5,        # スコア 3.5 以上でエントリー
        "exit_stop": "composite_negative",
        "exit_stop_threshold": -1.0,   # スコアが -1.0 を割ったら出口
        "stop_type": "atr",
        "target_type": "rr_ratio",
        "target_rr": 3.0,
        "max_holding_days": 30,
        "position_size_pct": 10.0,
        "description": (
            "RSI・MACD・BB・一目均衡表・OBV・トレンド系の6指標を"
            "加重スコアリングした複合シグナル戦略。"
            "スコアが閾値を上方ブレイクした時にエントリー。"
            "Pring(2002) のインターマーケット分析的アプローチ。"
        ),
        "reference": "Pring, M. (2002). Technical Analysis Explained.",
        "emoji": "🔮",
    },

    # ──────────────────────────────────────────
    # 6. 買鉄＋複合スコア（最強フィルター）
    # ──────────────────────────────────────────
    "買鉄＋複合（最強）": {
        "entry_condition": "kaitetsu_plus_composite",
        "kaitetsu_min": 4,             # 買鉄4条件以上
        "composite_min": 2.5,          # 複合スコア 2.5 以上
        "exit_stop": "ma5_break",
        "stop_type": "ma5",
        "target_type": "resistance",
        "max_holding_days": 30,
        "position_size_pct": 12.0,     # 確信度が高いのでサイズ大
        "description": (
            "Y式買鉄（4条件以上）と複合スコア指標（2.5以上）の"
            "ダブルフィルター戦略。トレード機会は少ないが、"
            "セットアップ品質が最も高い。ケリー基準的に最適なサイズ。"
        ),
        "reference": "Kelly, J.L. (1956). A New Interpretation of Information Rate.",
        "emoji": "🚀",
    },

    # ──────────────────────────────────────────
    # 7. BBスクイーズブレイク戦略
    # ──────────────────────────────────────────
    "BBスクイーズブレイク": {
        "entry_condition": "bb_squeeze_break",
        "exit_stop": "ma5_break",
        "stop_type": "atr",
        "target_type": "rr_ratio",
        "target_rr": 2.5,
        "max_holding_days": 20,
        "position_size_pct": 8.0,
        "description": (
            "Bollinger Band のスクイーズ（バンド幅収縮）後の"
            "上方ブレイクアウトを狙う戦略。"
            "ボラティリティの低下後に大きなムーブが来るという"
            "John Bollinger の理論に基づく。5MA 上向きで追加確認。"
        ),
        "reference": "Bollinger, J. (2001). Bollinger on Bollinger Bands.",
        "emoji": "💥",
    },
}


# ──────────────────────────────────────────────
# 戦略名リスト（UI用）
# ──────────────────────────────────────────────
STRATEGY_NAMES = list(STRATEGIES.keys())


def get_strategy(name: str) -> dict:
    """戦略名から設定dictを取得する"""
    if name not in STRATEGIES:
        raise ValueError(f"未定義の戦略: {name}\n有効: {STRATEGY_NAMES}")
    return STRATEGIES[name]


# ──────────────────────────────────────────────
# デフォルト銘柄リスト（バッチテスト用）
# ──────────────────────────────────────────────
DEFAULT_US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "AMD", "PLTR",
    "SMCI", "NET", "CRWD", "DDOG", "SNOW",
]

DEFAULT_JP_TICKERS = [
    "7203",  # トヨタ
    "6758",  # ソニー
    "9984",  # ソフトバンクG
    "6861",  # キーエンス
    "4063",  # 信越化学
    "8306",  # 三菱UFJ
    "7974",  # 任天堂
    "6920",  # レーザーテック
    "4519",  # 中外製薬
    "6098",  # リクルート
]
