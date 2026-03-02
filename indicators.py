"""
一般的に有効性が実証されているテクニカル指標モジュール

採用根拠（期待値ベース）:
- RSI: オシレーター系で最も実証されている平均回帰指標
- MACD: トレンドフォロー × モメンタムの複合（教科書的標準）
- Bollinger Bands: ボラティリティ・ブレイクアウト検出（スクイーズ戦略は有名）
- Ichimoku: 日本発。多くの機関投資家が参照する複合指標
- OBV: 価格より先行する出来高動向（Smart Moneyの追跡）
- ATR: ポジションサイズ・損切り幅の定量化（Van Tharpe推奨）
- Fibonacci: 大口投資家が意識する黄金比レベル
- Momentum/ROC: Jegadeesh & Titman (1993) で実証された価格継続性
- Golden/Death Cross: 50MA × 200MA クロス（長期トレンド変換）
- 52週高値ブレイク: アノマリー研究で実証された継続性シグナル
- 出来高急増: 機関投資家の参入サイン（大口の追跡）
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ────────────────────────────────────────────
# データクラス
# ────────────────────────────────────────────

@dataclass
class RSIResult:
    value: float = 0.0
    signal: str = "中立"     # "買われすぎ" | "売られすぎ" | "中立"
    divergence: str = "なし"  # "強気ダイバージェンス" | "弱気ダイバージェンス" | "なし"

    @property
    def color(self) -> str:
        if self.value >= 70:
            return "#FF4444"
        if self.value <= 30:
            return "#00CC66"
        return "#AAAAAA"

    @property
    def emoji(self) -> str:
        if self.value >= 80:
            return "🔴"
        if self.value >= 70:
            return "🟠"
        if self.value <= 20:
            return "🟢"
        if self.value <= 30:
            return "🟩"
        return "⚪"


@dataclass
class MACDResult:
    macd_line: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    cross_signal: str = "なし"   # "ゴールデンクロス" | "デッドクロス" | "なし"
    trend: str = "中立"           # "上昇" | "下落" | "中立"
    histogram_trend: str = "拡大中" # "拡大中" | "縮小中"


@dataclass
class BollingerResult:
    upper: float = 0.0
    middle: float = 0.0
    lower: float = 0.0
    width: float = 0.0        # バンド幅（ボラティリティ）
    pct_b: float = 0.5        # %B（0=下限, 1=上限）
    squeeze: bool = False     # スクイーズ（バンド収縮 → 大きな動きの前触れ）
    squeeze_release: bool = False  # スクイーズ解放（ブレイクアウト）
    walk_upper: bool = False  # バンドウォーク（上バンドに沿って上昇）
    walk_lower: bool = False  # バンドウォーク（下バンドに沿って下落）


@dataclass
class IchimokuResult:
    tenkan: float = 0.0          # 転換線 (9期間)
    kijun: float = 0.0           # 基準線 (26期間)
    senkou_a: float = 0.0        # 先行スパンA
    senkou_b: float = 0.0        # 先行スパンB
    chikou: float = 0.0          # 遅行スパン
    above_cloud: bool = False    # 株価が雲の上
    below_cloud: bool = False    # 株価が雲の下
    cloud_color: str = "neutral" # "bullish" | "bearish" | "neutral"
    tk_cross: str = "なし"       # "ゴールデン" | "デッド" | "なし"


@dataclass
class MomentumResult:
    roc_1m: float = 0.0    # 1ヶ月リターン
    roc_3m: float = 0.0    # 3ヶ月リターン（Jegadeesh & Titman）
    roc_6m: float = 0.0    # 6ヶ月リターン
    roc_12m: float = 0.0   # 12ヶ月リターン
    momentum_score: float = 0.0  # 総合モメンタムスコア（-1〜+1）
    rank_signal: str = "中立"    # "強い上昇モメンタム" | "弱い" | "下落モメンタム"


@dataclass
class VolumeAnalysisResult:
    obv: float = 0.0              # 最新OBV値
    obv_trend: str = "中立"       # "上昇" | "下落" | "中立"
    obv_divergence: str = "なし"  # "強気" | "弱気" | "なし"
    vol_ratio: float = 1.0        # 直近出来高 / 20日平均出来高
    is_high_volume: bool = False  # 出来高急増（2倍以上）
    volume_trend: str = "中立"    # "増加傾向" | "減少傾向" | "中立"


@dataclass
class FibonacciResult:
    swing_high: float = 0.0
    swing_low: float = 0.0
    direction: str = "上昇"  # "上昇" | "下落"
    levels: dict = field(default_factory=dict)  # {0.236: price, 0.382: price, ...}
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None


@dataclass
class CrossSignalResult:
    golden_cross_50_200: bool = False  # 50MA > 200MA（長期上昇）
    death_cross_50_200: bool = False   # 50MA < 200MA（長期下落）
    cross_date: Optional[str] = None
    days_since_cross: int = 0
    near_52w_high: bool = False    # 52週高値の95%以上（ブレイクアウト接近）
    broke_52w_high: bool = False   # 52週高値を超えた（実証済みアノマリー）


@dataclass
class ATRResult:
    atr_14: float = 0.0           # 14期間ATR
    atr_pct: float = 0.0          # ATR / 現在値（%）
    volatility_regime: str = "中" # "低" | "中" | "高"
    suggested_stop_atr: float = 0.0   # 推奨損切り（ATR×2）
    suggested_target_atr: float = 0.0 # 推奨目標（ATR×6 = 1:3 RR）


@dataclass
class CompositeSignal:
    """
    複数指標の統合スコアリングシステム
    研究ベースの重み付きスコアで総合判定
    """
    total_score: float = 0.0      # -10 〜 +10
    buy_signals: list = field(default_factory=list)
    sell_signals: list = field(default_factory=list)
    neutral_signals: list = field(default_factory=list)

    rsi_score: float = 0.0
    macd_score: float = 0.0
    bb_score: float = 0.0
    ichimoku_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    cross_score: float = 0.0

    @property
    def verdict(self) -> str:
        if self.total_score >= 5:
            return "強い買いシグナル"
        if self.total_score >= 2:
            return "弱い買いシグナル"
        if self.total_score <= -5:
            return "強い売りシグナル"
        if self.total_score <= -2:
            return "弱い売りシグナル"
        return "中立（様子見）"

    @property
    def color(self) -> str:
        if self.total_score >= 3:
            return "#00CC66"
        if self.total_score >= 1:
            return "#66DD88"
        if self.total_score <= -3:
            return "#FF3333"
        if self.total_score <= -1:
            return "#FF8888"
        return "#AAAAAA"

    @property
    def emoji(self) -> str:
        if self.total_score >= 5:
            return "🚀"
        if self.total_score >= 2:
            return "📈"
        if self.total_score <= -5:
            return "💀"
        if self.total_score <= -2:
            return "📉"
        return "➡️"


# ────────────────────────────────────────────
# RSI
# ────────────────────────────────────────────

def calc_rsi(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, RSIResult]:
    """
    RSI（相対力指数）の計算
    - >70: 買われすぎ圏（慎重）
    - <30: 売られすぎ圏（反発候補）
    - ダイバージェンス: 価格とRSIの乖離
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series = rsi_series.fillna(50)

    result = RSIResult()
    if len(rsi_series) < period:
        return rsi_series, result

    result.value = float(rsi_series.iloc[-1])

    if result.value >= 70:
        result.signal = "買われすぎ"
    elif result.value <= 30:
        result.signal = "売られすぎ"
    else:
        result.signal = "中立"

    # ダイバージェンス検出（直近10〜30本）
    lookback = min(30, len(df) - 1)
    prices = df["Close"].iloc[-lookback:].values
    rsi_vals = rsi_series.iloc[-lookback:].values

    price_trend = np.polyfit(range(lookback), prices, 1)[0]
    rsi_trend = np.polyfit(range(lookback), rsi_vals, 1)[0]

    # 弱気ダイバージェンス: 価格↑ RSI↓ → 上昇の勢い鈍化
    if price_trend > 0 and rsi_trend < -0.3:
        result.divergence = "弱気ダイバージェンス（上昇勢い鈍化）"
    # 強気ダイバージェンス: 価格↓ RSI↑ → 下落の勢い鈍化
    elif price_trend < 0 and rsi_trend > 0.3:
        result.divergence = "強気ダイバージェンス（下落勢い鈍化）"

    return rsi_series, result


# ────────────────────────────────────────────
# MACD
# ────────────────────────────────────────────

def calc_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series, MACDResult]:
    """
    MACD（移動平均収束拡散法）
    - MACD > Signal: 上昇モメンタム
    - ゴールデンクロス: 強い買いシグナル
    - ヒストグラム縮小: モメンタム弱まり
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result = MACDResult()
    if len(macd_line) < slow + signal:
        return macd_line, signal_line, histogram, result

    result.macd_line = float(macd_line.iloc[-1])
    result.signal_line = float(signal_line.iloc[-1])
    result.histogram = float(histogram.iloc[-1])

    # クロスシグナル
    prev_macd = float(macd_line.iloc[-2])
    prev_signal = float(signal_line.iloc[-2])
    curr_macd = result.macd_line
    curr_signal = result.signal_line

    if prev_macd < prev_signal and curr_macd > curr_signal:
        result.cross_signal = "ゴールデンクロス"
    elif prev_macd > prev_signal and curr_macd < curr_signal:
        result.cross_signal = "デッドクロス"

    result.trend = "上昇" if curr_macd > curr_signal else "下落"

    # ヒストグラムトレンド
    prev_hist = float(histogram.iloc[-2])
    curr_hist = result.histogram
    if abs(curr_hist) > abs(prev_hist):
        result.histogram_trend = "拡大中（モメンタム強化）"
    else:
        result.histogram_trend = "縮小中（モメンタム弱体化）"

    return macd_line, signal_line, histogram, result


# ────────────────────────────────────────────
# Bollinger Bands
# ────────────────────────────────────────────

def calc_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, BollingerResult]:
    """
    ボリンジャーバンド
    - スクイーズ: バンド幅が収縮 → 大きな動きの前触れ（John Bollinger推奨戦略）
    - バンドウォーク: バンドに沿って動く強いトレンド
    - %B: バンド内での位置（0=下限, 1=上限）
    """
    middle = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    result = BollingerResult()
    if len(middle) < period or pd.isna(middle.iloc[-1]):
        return upper, middle, lower, result

    result.upper = float(upper.iloc[-1])
    result.middle = float(middle.iloc[-1])
    result.lower = float(lower.iloc[-1])

    band_range = result.upper - result.lower
    result.width = band_range / result.middle if result.middle > 0 else 0

    close = float(df["Close"].iloc[-1])
    result.pct_b = (close - result.lower) / band_range if band_range > 0 else 0.5

    # スクイーズ判定: 直近バンド幅が過去50日の最小値付近
    lookback = min(50, len(df))
    widths = (upper - lower) / middle
    recent_widths = widths.iloc[-lookback:]
    min_width = recent_widths.min()
    current_width = widths.iloc[-1]

    if not pd.isna(current_width) and not pd.isna(min_width):
        result.squeeze = float(current_width) < float(min_width) * 1.05

        # スクイーズ解放: 前日スクイーズだったが今日拡大
        prev_width = float(widths.iloc[-2]) if len(widths) > 1 else 0
        result.squeeze_release = (
            prev_width < float(min_width) * 1.05
            and float(current_width) > float(min_width) * 1.1
        )

    # バンドウォーク判定（直近5本）
    last5_close = df["Close"].iloc[-5:].values
    last5_upper = upper.iloc[-5:].values
    last5_lower = lower.iloc[-5:].values

    walk_up_count = sum(1 for i in range(5) if last5_close[i] >= last5_upper[i] * 0.99)
    walk_down_count = sum(1 for i in range(5) if last5_close[i] <= last5_lower[i] * 1.01)
    result.walk_upper = walk_up_count >= 3
    result.walk_lower = walk_down_count >= 3

    return upper, middle, lower, result


# ────────────────────────────────────────────
# 一目均衡表 (Ichimoku Cloud)
# ────────────────────────────────────────────

def calc_ichimoku(df: pd.DataFrame) -> tuple[dict, IchimokuResult]:
    """
    一目均衡表
    - 転換線: 9期間高値安値の中値
    - 基準線: 26期間高値安値の中値（トレンドの重心）
    - 先行スパンA・B: 未来26期間に表示される「雲」
    - 遅行スパン: 26期間過去の終値（確認線）

    判定基準:
    - 株価が雲の上 = 強気
    - TK クロス（転換線が基準線を上抜け）= 買いシグナル
    - 三役好転（全条件が揃う）= 最強の買いシグナル
    """
    def midpoint(high, low, n):
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan = midpoint(df["High"], df["Low"], 9)
    kijun = midpoint(df["High"], df["Low"], 26)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = midpoint(df["High"], df["Low"], 52).shift(26)
    chikou = df["Close"].shift(-26)

    lines = {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }

    result = IchimokuResult()
    if len(df) < 52:
        return lines, result

    result.tenkan = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else 0
    result.kijun = float(kijun.iloc[-1]) if not pd.isna(kijun.iloc[-1]) else 0
    result.senkou_a = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else 0
    result.senkou_b = float(senkou_b.iloc[-1]) if not pd.isna(senkou_b.iloc[-1]) else 0

    close = float(df["Close"].iloc[-1])
    cloud_top = max(result.senkou_a, result.senkou_b)
    cloud_bottom = min(result.senkou_a, result.senkou_b)

    result.above_cloud = close > cloud_top
    result.below_cloud = close < cloud_bottom

    # 雲の色（先行スパンAがBより上 = 緑雲 = 強気）
    if result.senkou_a > result.senkou_b:
        result.cloud_color = "bullish"
    elif result.senkou_a < result.senkou_b:
        result.cloud_color = "bearish"
    else:
        result.cloud_color = "neutral"

    # TKクロス（転換線 vs 基準線）
    prev_tenkan = float(tenkan.iloc[-2]) if not pd.isna(tenkan.iloc[-2]) else 0
    prev_kijun = float(kijun.iloc[-2]) if not pd.isna(kijun.iloc[-2]) else 0

    if prev_tenkan < prev_kijun and result.tenkan > result.kijun:
        result.tk_cross = "ゴールデン（転換線が基準線を上抜け）"
    elif prev_tenkan > prev_kijun and result.tenkan < result.kijun:
        result.tk_cross = "デッド（転換線が基準線を下抜け）"

    return lines, result


# ────────────────────────────────────────────
# OBV (On-Balance Volume)
# ────────────────────────────────────────────

def calc_obv(df: pd.DataFrame) -> tuple[pd.Series, VolumeAnalysisResult]:
    """
    OBV（累積出来高）+ 出来高分析
    - OBVの上昇 = Smart Moneyが買い集め中
    - 価格上昇 + OBV下落 = 弱気ダイバージェンス（注意）
    - 価格下落 + OBV上昇 = 強気ダイバージェンス（底入れ候補）
    - 出来高急増（2倍以上）= 機関投資家の参入サイン
    """
    direction = np.where(df["Close"].diff() > 0, 1,
                np.where(df["Close"].diff() < 0, -1, 0))
    obv = (direction * df["Volume"]).cumsum()

    result = VolumeAnalysisResult()
    if len(obv) < 20:
        return obv, result

    result.obv = float(obv.iloc[-1])

    # OBVトレンド（直近20本の傾き）
    lookback = min(20, len(obv))
    obv_recent = obv.iloc[-lookback:].values
    obv_slope = np.polyfit(range(lookback), obv_recent, 1)[0]
    result.obv_trend = "上昇" if obv_slope > 0 else "下落" if obv_slope < 0 else "中立"

    # OBVダイバージェンス
    price_recent = df["Close"].iloc[-lookback:].values
    price_slope = np.polyfit(range(lookback), price_recent, 1)[0]
    if price_slope > 0 and obv_slope < 0:
        result.obv_divergence = "弱気（価格↑ OBV↓）"
    elif price_slope < 0 and obv_slope > 0:
        result.obv_divergence = "強気（価格↓ OBV↑）"

    # 出来高分析
    avg_vol = float(df["Volume"].iloc[-20:].mean())
    latest_vol = float(df["Volume"].iloc[-1])
    result.vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    result.is_high_volume = result.vol_ratio >= 2.0

    # 出来高トレンド（増加 or 減少）
    vol_5d = float(df["Volume"].iloc[-5:].mean())
    vol_20d = float(df["Volume"].iloc[-20:].mean())
    result.volume_trend = "増加傾向" if vol_5d > vol_20d * 1.1 else "減少傾向" if vol_5d < vol_20d * 0.9 else "中立"

    return obv, result


# ────────────────────────────────────────────
# ATR (Average True Range)
# ────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, ATRResult]:
    """
    ATR（平均真値レンジ）
    - ポジションサイズ計算の基準（Van Tharpe の「1R」概念）
    - ATR×2 が合理的な損切り幅
    - ATR×6 が1:3リスクリワードの目標
    - ATR > 3% = 高ボラティリティ（慎重に）
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_series = tr.ewm(span=period, adjust=False).mean()

    result = ATRResult()
    if len(atr_series) < period or pd.isna(atr_series.iloc[-1]):
        return atr_series, result

    result.atr_14 = float(atr_series.iloc[-1])
    close = float(df["Close"].iloc[-1])
    result.atr_pct = result.atr_14 / close * 100 if close > 0 else 0

    if result.atr_pct < 1.5:
        result.volatility_regime = "低"
    elif result.atr_pct > 4.0:
        result.volatility_regime = "高"
    else:
        result.volatility_regime = "中"

    result.suggested_stop_atr = close - result.atr_14 * 2
    result.suggested_target_atr = close + result.atr_14 * 6  # 1:3 RR

    return atr_series, result


# ────────────────────────────────────────────
# フィボナッチ・リトレースメント
# ────────────────────────────────────────────

def calc_fibonacci(df: pd.DataFrame, lookback: int = 60) -> FibonacciResult:
    """
    フィボナッチ・リトレースメント
    - 大口投資家が意識する黄金比の価格レベル
    - 重要レベル: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    - 61.8% は「黄金比」で最も強い反転レベル

    直近の最高値・最安値を基準に自動算出
    """
    result = FibonacciResult()
    recent = df.iloc[-min(lookback, len(df)):]

    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    high_idx = recent["High"].idxmax()
    low_idx = recent["Low"].idxmin()

    result.swing_high = swing_high
    result.swing_low = swing_low

    # 方向: 高値が安値より後に来たら「下落」
    if df.index.get_loc(high_idx) > df.index.get_loc(low_idx):
        result.direction = "下落"
        diff = swing_high - swing_low
        for ratio in [0.236, 0.382, 0.500, 0.618, 0.786]:
            result.levels[ratio] = swing_high - diff * ratio
    else:
        result.direction = "上昇"
        diff = swing_high - swing_low
        for ratio in [0.236, 0.382, 0.500, 0.618, 0.786]:
            result.levels[ratio] = swing_low + diff * ratio

    # 現在値に最も近い支持・抵抗
    close = float(df["Close"].iloc[-1])
    fib_levels = sorted(result.levels.values())
    above = [p for p in fib_levels if p > close]
    below = [p for p in fib_levels if p < close]
    result.nearest_resistance = min(above) if above else None
    result.nearest_support = max(below) if below else None

    return result


# ────────────────────────────────────────────
# モメンタム / Rate of Change
# ────────────────────────────────────────────

def calc_momentum(df: pd.DataFrame) -> MomentumResult:
    """
    モメンタム / ROC（変化率）
    - Jegadeesh & Titman (1993): 3〜12ヶ月のリターンは継続する傾向
    - 特に6ヶ月・12ヶ月リターンが強い銘柄は今後1〜3ヶ月継続しやすい
    - 直近1ヶ月リターンは短期的に逆転する傾向（平均回帰）
    """
    result = MomentumResult()
    close = df["Close"]
    n = len(close)

    def roc(days):
        if n > days:
            v0 = float(close.iloc[-(days + 1)])
            v1 = float(close.iloc[-1])
            return (v1 - v0) / v0 * 100 if v0 > 0 else 0.0
        return 0.0

    result.roc_1m = roc(21)
    result.roc_3m = roc(63)
    result.roc_6m = roc(126)
    result.roc_12m = roc(252)

    # 総合モメンタムスコア（6M・12M に重点）
    score = 0.0
    weight_total = 0.0
    if n > 63:
        score += result.roc_3m * 0.2
        weight_total += 0.2
    if n > 126:
        score += result.roc_6m * 0.4
        weight_total += 0.4
    if n > 252:
        score += result.roc_12m * 0.4
        weight_total += 0.4

    if weight_total > 0:
        avg = score / weight_total
        result.momentum_score = max(-1.0, min(1.0, avg / 30))  # ±30%を±1に正規化

    if result.momentum_score > 0.3:
        result.rank_signal = "強い上昇モメンタム"
    elif result.momentum_score < -0.3:
        result.rank_signal = "強い下落モメンタム"
    else:
        result.rank_signal = "モメンタム弱い（様子見）"

    return result


# ────────────────────────────────────────────
# ゴールデン/デッドクロス & 52週高値
# ────────────────────────────────────────────

def calc_cross_signals(df: pd.DataFrame) -> CrossSignalResult:
    """
    長期クロスシグナル + 52週高値ブレイクアウト
    - Golden Cross (50MA/200MA): 長期上昇トレンドの始まりを示す
    - 52週高値ブレイク: アノマリー研究で実証されたモメンタムシグナル
      （George & Hwang 2004: 52週高値付近は機関投資家の注文集中帯）
    """
    result = CrossSignalResult()
    if len(df) < 200:
        return result

    ma50 = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    curr_50 = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else 0
    curr_200 = float(ma200.iloc[-1]) if not pd.isna(ma200.iloc[-1]) else 0
    prev_50 = float(ma50.iloc[-2]) if not pd.isna(ma50.iloc[-2]) else 0
    prev_200 = float(ma200.iloc[-2]) if not pd.isna(ma200.iloc[-2]) else 0

    if prev_50 < prev_200 and curr_50 > curr_200:
        result.golden_cross_50_200 = True
        result.cross_date = str(df.index[-1].date())
        result.days_since_cross = 0
    elif prev_50 > prev_200 and curr_50 < curr_200:
        result.death_cross_50_200 = True

    # 直近でクロスが起きた日を探す（過去60日）
    for i in range(2, min(60, len(df))):
        ma50_i = float(ma50.iloc[-i]) if not pd.isna(ma50.iloc[-i]) else 0
        ma200_i = float(ma200.iloc[-i]) if not pd.isna(ma200.iloc[-i]) else 0
        ma50_prev = float(ma50.iloc[-i - 1]) if not pd.isna(ma50.iloc[-i - 1]) else 0
        ma200_prev = float(ma200.iloc[-i - 1]) if not pd.isna(ma200.iloc[-i - 1]) else 0
        if (ma50_prev < ma200_prev and ma50_i > ma200_i) or \
           (ma50_prev > ma200_prev and ma50_i < ma200_i):
            result.days_since_cross = i
            break

    # 52週高値
    close = float(df["Close"].iloc[-1])
    high_52w = float(df["High"].iloc[-min(252, len(df)):].max())
    result.near_52w_high = close >= high_52w * 0.95
    result.broke_52w_high = close >= high_52w

    return result


# ────────────────────────────────────────────
# 統合スコアリング
# ────────────────────────────────────────────

def calc_composite_signal(
    rsi_r: RSIResult,
    macd_r: MACDResult,
    bb_r: BollingerResult,
    ichi_r: IchimokuResult,
    mom_r: MomentumResult,
    vol_r: VolumeAnalysisResult,
    cross_r: CrossSignalResult,
) -> CompositeSignal:
    """
    各指標を重み付けして総合スコアを算出

    スコア範囲: -10 〜 +10
    - 正 = 買い優位
    - 負 = 売り優位
    - ±3以上で有意なシグナル

    重み付けの考え方:
    - トレンドフォロー系（Ichimoku/クロス）を重視（再現性が高い）
    - モメンタム（学術研究で実証）を重視
    - オシレーター系（RSI/BB）は補助的に使用
    """
    sig = CompositeSignal()

    # ── RSI スコア ──────────────────────────
    if rsi_r.value <= 25:
        sig.rsi_score = 2.0
        sig.buy_signals.append(f"RSI極度売られすぎ ({rsi_r.value:.0f})")
    elif rsi_r.value <= 35:
        sig.rsi_score = 1.0
        sig.buy_signals.append(f"RSI売られすぎ ({rsi_r.value:.0f})")
    elif rsi_r.value >= 75:
        sig.rsi_score = -2.0
        sig.sell_signals.append(f"RSI極度買われすぎ ({rsi_r.value:.0f})")
    elif rsi_r.value >= 65:
        sig.rsi_score = -1.0
        sig.sell_signals.append(f"RSI買われすぎ ({rsi_r.value:.0f})")
    else:
        sig.neutral_signals.append(f"RSI中立 ({rsi_r.value:.0f})")

    if "強気ダイバージェンス" in rsi_r.divergence:
        sig.rsi_score += 1.0
        sig.buy_signals.append("RSI強気ダイバージェンス")
    elif "弱気ダイバージェンス" in rsi_r.divergence:
        sig.rsi_score -= 1.0
        sig.sell_signals.append("RSI弱気ダイバージェンス")

    # ── MACD スコア ─────────────────────────
    if macd_r.cross_signal == "ゴールデンクロス":
        sig.macd_score = 2.0
        sig.buy_signals.append("MACDゴールデンクロス")
    elif macd_r.cross_signal == "デッドクロス":
        sig.macd_score = -2.0
        sig.sell_signals.append("MACDデッドクロス")
    elif macd_r.trend == "上昇":
        sig.macd_score = 0.5
        sig.buy_signals.append("MACD上昇トレンド")
    else:
        sig.macd_score = -0.5
        sig.sell_signals.append("MACD下落トレンド")

    if "拡大" in macd_r.histogram_trend and macd_r.histogram > 0:
        sig.macd_score += 0.5
    elif "縮小" in macd_r.histogram_trend and macd_r.histogram > 0:
        sig.macd_score -= 0.5

    # ── BB スコア ───────────────────────────
    if bb_r.squeeze:
        sig.bb_score = 0.5
        sig.neutral_signals.append("BBスクイーズ（ブレイク待ち）")
    if bb_r.squeeze_release:
        sig.bb_score = 1.5
        sig.buy_signals.append("BBスクイーズ解放（ブレイクアウト！）")
    if bb_r.pct_b < 0.1:
        sig.bb_score += 1.0
        sig.buy_signals.append(f"BB下限タッチ（%B={bb_r.pct_b:.2f}）")
    elif bb_r.pct_b > 0.9:
        sig.bb_score -= 0.5
        sig.sell_signals.append(f"BB上限タッチ（%B={bb_r.pct_b:.2f}）")
    if bb_r.walk_upper:
        sig.bb_score += 0.5
        sig.buy_signals.append("BBバンドウォーク（強いトレンド）")
    if bb_r.walk_lower:
        sig.bb_score -= 0.5
        sig.sell_signals.append("BB下バンドウォーク（強い下落）")

    # ── 一目均衡表 スコア ────────────────────
    if ichi_r.above_cloud:
        sig.ichimoku_score = 1.5
        sig.buy_signals.append("株価が一目の雲の上（強気）")
    elif ichi_r.below_cloud:
        sig.ichimoku_score = -1.5
        sig.sell_signals.append("株価が一目の雲の下（弱気）")
    else:
        sig.neutral_signals.append("株価が一目の雲の中（中立）")

    if ichi_r.cloud_color == "bullish":
        sig.ichimoku_score += 0.5
        sig.buy_signals.append("一目の雲が緑（先行A>先行B）")
    elif ichi_r.cloud_color == "bearish":
        sig.ichimoku_score -= 0.5
        sig.sell_signals.append("一目の雲が赤（先行B>先行A）")

    if "ゴールデン" in ichi_r.tk_cross:
        sig.ichimoku_score += 1.0
        sig.buy_signals.append("一目TKクロス（転換線↑基準線）")
    elif "デッド" in ichi_r.tk_cross:
        sig.ichimoku_score -= 1.0
        sig.sell_signals.append("一目TKデッドクロス")

    # ── モメンタム スコア ────────────────────
    sig.momentum_score = mom_r.momentum_score * 2.5  # -2.5〜+2.5
    if mom_r.momentum_score > 0.3:
        sig.buy_signals.append(f"強い上昇モメンタム（6M: {mom_r.roc_6m:+.1f}%）")
    elif mom_r.momentum_score < -0.3:
        sig.sell_signals.append(f"強い下落モメンタム（6M: {mom_r.roc_6m:+.1f}%）")
    else:
        sig.neutral_signals.append(f"モメンタム弱い（6M: {mom_r.roc_6m:+.1f}%）")

    # ── 出来高 スコア ────────────────────────
    if vol_r.obv_trend == "上昇":
        sig.volume_score = 0.5
        sig.buy_signals.append("OBV上昇（Smart Money買い集め）")
    elif vol_r.obv_trend == "下落":
        sig.volume_score = -0.5
        sig.sell_signals.append("OBV下落（Smart Money売り）")

    if "強気" in vol_r.obv_divergence:
        sig.volume_score += 1.0
        sig.buy_signals.append("OBV強気ダイバージェンス（底入れ候補）")
    elif "弱気" in vol_r.obv_divergence:
        sig.volume_score -= 1.0
        sig.sell_signals.append("OBV弱気ダイバージェンス（注意）")

    if vol_r.is_high_volume:
        # 高出来高は方向に応じて加点/減点
        sig.neutral_signals.append(f"出来高急増（{vol_r.vol_ratio:.1f}x平均）")

    # ── ゴールデン/デッドクロス & 52週 スコア ─
    if cross_r.golden_cross_50_200:
        sig.cross_score = 2.0
        sig.buy_signals.append("50MA/200MAゴールデンクロス（長期上昇開始）")
    elif cross_r.death_cross_50_200:
        sig.cross_score = -2.0
        sig.sell_signals.append("50MA/200MAデッドクロス（長期下落開始）")
    elif cross_r.days_since_cross > 0:
        # クロス後の継続シグナル
        from_cross_days = cross_r.days_since_cross
        if from_cross_days < 30:
            sig.cross_score = 1.0
            sig.buy_signals.append(f"ゴールデンクロス後{from_cross_days}日（継続期待）")

    if cross_r.broke_52w_high:
        sig.cross_score += 1.5
        sig.buy_signals.append("52週高値ブレイク（アノマリー：継続性高い）")
    elif cross_r.near_52w_high:
        sig.cross_score += 0.5
        sig.buy_signals.append("52週高値に接近（ブレイク待ち）")

    # ── 合計スコア ───────────────────────────
    sig.total_score = (
        sig.rsi_score
        + sig.macd_score
        + sig.bb_score
        + sig.ichimoku_score
        + sig.momentum_score
        + sig.volume_score
        + sig.cross_score
    )
    # クランプ
    sig.total_score = max(-10.0, min(10.0, sig.total_score))

    return sig


# ────────────────────────────────────────────
# 全指標を一括計算
# ────────────────────────────────────────────

def run_all_indicators(df: pd.DataFrame) -> dict:
    """全指標を計算して辞書で返す"""
    rsi_series, rsi_r = calc_rsi(df)
    macd_line, signal_line, histogram, macd_r = calc_macd(df)
    bb_upper, bb_mid, bb_lower, bb_r = calc_bollinger(df)
    ichi_lines, ichi_r = calc_ichimoku(df)
    mom_r = calc_momentum(df)
    obv_series, vol_r = calc_obv(df)
    atr_series, atr_r = calc_atr(df)
    fib_r = calc_fibonacci(df)
    cross_r = calc_cross_signals(df)

    composite = calc_composite_signal(rsi_r, macd_r, bb_r, ichi_r, mom_r, vol_r, cross_r)

    return {
        "rsi_series": rsi_series,
        "rsi": rsi_r,
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
        "macd": macd_r,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb": bb_r,
        "ichi_lines": ichi_lines,
        "ichi": ichi_r,
        "momentum": mom_r,
        "obv_series": obv_series,
        "volume": vol_r,
        "atr_series": atr_series,
        "atr": atr_r,
        "fibonacci": fib_r,
        "cross": cross_r,
        "composite": composite,
    }
