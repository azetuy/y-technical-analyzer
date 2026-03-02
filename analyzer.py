"""
よろち式テクニカル分析エンジン
株価データを取得し、よろちさん講義に基づく各種分析を行う
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# 一般テクニカル指標（indicators.py）は run_full_analysis 内で遅延インポート


# ────────────────────────────────────────────
# データクラス
# ────────────────────────────────────────────

@dataclass
class KaitetsuResult:
    """買鉄（かいてつ）5条件チェック結果"""
    cond1_ma_order: bool = False      # 5MA > 20MA
    cond2_ma20_up: bool = False       # 20MAが上向き
    cond3_ma5_turn_up: bool = False   # 5MAが下→上に転換
    cond4_body_above_ma5: bool = False  # 実体の50%以上が5MA上
    cond5_near_ma20: bool = False     # 20MAに接近/タッチ

    @property
    def score(self) -> int:
        return sum([self.cond1_ma_order, self.cond2_ma20_up,
                    self.cond3_ma5_turn_up, self.cond4_body_above_ma5,
                    self.cond5_near_ma20])

    @property
    def is_valid(self) -> bool:
        return self.score == 5

    @property
    def label(self) -> str:
        if self.score == 5:
            return "✅ 買鉄シグナル！"
        elif self.score >= 3:
            return f"⚠️ 買鉄候補 ({self.score}/5条件)"
        else:
            return f"❌ 買鉄非該当 ({self.score}/5条件)"


@dataclass
class TrendResult:
    """トレンド判定結果"""
    phase: str = "不明"          # "上昇" | "横ばい" | "下落"
    ma400_up: bool = False       # 400日線が上向き
    price_above_ma400: bool = False  # 株価が400日線上
    perfect_order: str = "なし"  # "最強" | "監視" | "エントリー" | "最弱" | "なし"
    ma5_slope: float = 0.0
    ma20_slope: float = 0.0
    ma60_slope: float = 0.0
    days_above_ma20: int = 0     # 20MA上に乗ってから何日経過

    @property
    def emoji(self) -> str:
        return {"上昇": "🟢", "横ばい": "🟡", "下落": "🔴"}.get(self.phase, "⚪")

    @property
    def cycle_position(self) -> str:
        """トレンドサイクル上の位置を返す"""
        return self.phase


@dataclass
class SupportResistanceLevel:
    """支持線・抵抗線の1本"""
    price: float
    level_type: str       # "resistance" | "support"
    touch_count: int = 1
    strength: str = "弱"  # "強" | "中" | "弱"
    source: str = ""      # "高値安値" | "ギャップ" | "毛抜き" | "横ばい帯"

    @property
    def color(self) -> str:
        if self.level_type == "resistance":
            return "#FF4444" if self.touch_count >= 3 else "#FF8888"
        else:
            return "#4444FF" if self.touch_count >= 3 else "#8888FF"

    @property
    def width(self) -> int:
        if self.touch_count >= 3:
            return 2
        return 1


@dataclass
class GapInfo:
    """ギャップ（窓）情報"""
    date: pd.Timestamp
    gap_type: str         # "up" | "down"
    upper: float
    lower: float
    is_filled: bool = False


@dataclass
class PatternResult:
    """チャートパターン検出結果"""
    name: str
    detected: bool
    description: str = ""
    start_idx: int = -1
    end_idx: int = -1
    confidence: float = 0.0


@dataclass
class AnalysisResult:
    """分析結果の統合"""
    ticker: str
    df: pd.DataFrame
    trend: TrendResult
    kaitetsu: KaitetsuResult
    support_resistance: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    high_low_marks: dict = field(default_factory=dict)  # よろち式高値安値マーク
    stop_loss_price: float = 0.0   # 損切り価格（現在の5MA）
    current_price: float = 0.0
    company_name: str = ""
    indicators: dict = field(default_factory=dict)      # 一般テクニカル指標


# ────────────────────────────────────────────
# データ取得
# ────────────────────────────────────────────

def get_ticker_symbol(code: str) -> str:
    """
    入力コードをYahoo Finance形式に変換
    - 数字4〜5桁 → 日本株（.T付加）
    - それ以外 → そのまま（米国株）
    """
    code = code.strip().upper()
    # 数字のみで構成（日本株証券コード）
    if code.isdigit() and len(code) in (4, 5):
        return f"{code}.T"
    # 既に.Tが付いている
    if code.endswith(".T"):
        return code
    return code


def fetch_stock_data(
    ticker_input: str,
    period: str = "1y",
    interval: str = "1d"
) -> tuple[Optional[pd.DataFrame], str, str]:
    """
    株価データを取得して前処理する

    Returns:
        (DataFrame, ticker_symbol, company_name)
    """
    symbol = get_ticker_symbol(ticker_input)
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return None, symbol, ""

        df.index = pd.to_datetime(df.index)
        # タイムゾーン除去（プロット用）
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # 移動平均線を追加
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["MA400"] = df["Close"].rolling(400).mean()

        # MA傾き（5期間の変化率）
        df["MA5_slope"] = df["MA5"].diff(3)
        df["MA20_slope"] = df["MA20"].diff(5)
        df["MA60_slope"] = df["MA60"].diff(10)
        df["MA400_slope"] = df["MA400"].diff(20)

        # 5MA方向（True=上向き, False=下向き）
        df["MA5_up"] = df["MA5_slope"] > 0

        # tk.info はレートリミットで失敗しやすいため個別にtry
        try:
            company_name = tk.info.get("longName", ticker_input)
        except Exception:
            company_name = ticker_input

        return df, symbol, company_name

    except Exception as e:
        print(f"データ取得エラー: {e}")
        return None, symbol, ""


# ────────────────────────────────────────────
# トレンド分析
# ────────────────────────────────────────────

def analyze_trend(df: pd.DataFrame) -> TrendResult:
    """よろち式トレンド判定"""
    result = TrendResult()

    if len(df) < 60:
        return result

    last = df.iloc[-1]
    prev = df.iloc[-5]  # 5日前と比較

    # 400日線の傾きと位置
    if not pd.isna(last.get("MA400")):
        result.ma400_up = float(df["MA400_slope"].iloc[-1]) > 0
        result.price_above_ma400 = float(last["Close"]) > float(last["MA400"])

    # MAの傾き
    result.ma5_slope = float(df["MA5_slope"].iloc[-1]) if not pd.isna(df["MA5_slope"].iloc[-1]) else 0
    result.ma20_slope = float(df["MA20_slope"].iloc[-1]) if not pd.isna(df["MA20_slope"].iloc[-1]) else 0
    result.ma60_slope = float(df["MA60_slope"].iloc[-1]) if not pd.isna(df["MA60_slope"].iloc[-1]) else 0

    # トレンドフェーズ判定
    ma5_up = result.ma5_slope > 0
    ma20_up = result.ma20_slope > 0
    ma60_up = result.ma60_slope > 0

    if result.price_above_ma400 and result.ma400_up and ma20_up:
        result.phase = "上昇"
    elif not result.price_above_ma400 and not result.ma400_up and not ma20_up:
        result.phase = "下落"
    else:
        result.phase = "横ばい"

    # パーフェクトオーダー判定
    ma5_val = float(last.get("MA5", 0) or 0)
    ma20_val = float(last.get("MA20", 0) or 0)
    ma60_val = float(last.get("MA60", 0) or 0)

    if ma5_val > 0 and ma20_val > 0 and ma60_val > 0:
        if ma5_up and ma20_up and ma60_up and ma5_val > ma20_val > ma60_val:
            result.perfect_order = "最強（パーフェクトオーダー）"
        elif not ma5_up and ma20_up and ma60_up:
            result.perfect_order = "監視（押し目候補）"
        elif ma5_up and ma20_up and ma60_up and not (ma5_val > ma20_val):
            result.perfect_order = "エントリー候補"
        elif not ma5_up and not ma20_up and not ma60_up:
            result.perfect_order = "最弱（全下向き）"

    # 20MA上に乗ってから何日経過
    days_count = 0
    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        if not pd.isna(row.get("MA20")) and float(row["Close"]) > float(row["MA20"]):
            days_count += 1
        else:
            break
    result.days_above_ma20 = days_count

    result.current_price = float(last["Close"])
    return result


# ────────────────────────────────────────────
# 買鉄分析
# ────────────────────────────────────────────

def analyze_kaitetsu(df: pd.DataFrame) -> KaitetsuResult:
    """よろち式 買鉄5条件を最新足でチェック"""
    result = KaitetsuResult()

    if len(df) < 10:
        return result

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ma5 = float(last.get("MA5") or 0)
    ma20 = float(last.get("MA20") or 0)
    if ma5 == 0 or ma20 == 0:
        return result

    # 条件1: 5MA > 20MA
    result.cond1_ma_order = ma5 > ma20

    # 条件2: 20MAが上向き
    result.cond2_ma20_up = float(df["MA20_slope"].iloc[-1]) > 0

    # 条件3: 5MAが下→上に転換（前日下向き→今日上向き）
    ma5_up_now = float(df["MA5_slope"].iloc[-1]) > 0
    ma5_up_prev = float(df["MA5_slope"].iloc[-2]) > 0 if len(df) > 2 else False
    result.cond3_ma5_turn_up = (not ma5_up_prev) and ma5_up_now

    # 条件4: 実体が5MAの上に50%以上
    open_p = float(last["Open"])
    close_p = float(last["Close"])
    body_bottom = min(open_p, close_p)
    body_top = max(open_p, close_p)
    body_len = body_top - body_bottom

    if body_len > 0:
        # 実体のうちMA5の上の部分の割合
        above_ma5_portion = max(0, body_top - ma5) / body_len
        result.cond4_body_above_ma5 = above_ma5_portion >= 0.5 and close_p > ma5
    else:
        result.cond4_body_above_ma5 = close_p > ma5

    # 条件5: 直近3〜10本で20MAに接近/タッチ（調整の証明）
    lookback = min(10, len(df))
    touched_ma20 = False
    for i in range(2, lookback):
        row = df.iloc[-i]
        low_p = float(row["Low"])
        ma20_i = float(row.get("MA20") or 0)
        if ma20_i > 0:
            dist_pct = abs(low_p - ma20_i) / ma20_i
            if dist_pct < 0.03:  # 3%以内に接近
                touched_ma20 = True
                break
    result.cond5_near_ma20 = touched_ma20

    return result


# ────────────────────────────────────────────
# よろち式 高値安値の抽出
# ────────────────────────────────────────────

def extract_yorochi_highs_lows(df: pd.DataFrame) -> tuple[list, list]:
    """
    よろち式定義に基づく高値安値を抽出

    高値: 5MAが「下→上」変化区間の最高値（ヒゲ含む）
    安値: 5MAが「上→下」変化区間の最低値（ヒゲ含む）

    Returns:
        (highs, lows) - 各々 (date, price) のリスト
    """
    if len(df) < 10:
        return [], []

    ma5_up = df["MA5_up"].tolist()
    highs = []
    lows = []

    i = 1
    while i < len(df):
        # 5MAが下→上に変化した区間（上向き区間の高値を探す）
        if not ma5_up[i - 1] and ma5_up[i]:
            # 上向き区間の開始
            start = i
            while i < len(df) and ma5_up[i]:
                i += 1
            end = i - 1
            # その区間の最高値（ヒゲ含む）
            segment = df.iloc[start:end + 1]
            if len(segment) > 0:
                max_idx = segment["High"].idxmax()
                highs.append((max_idx, float(segment["High"].max())))
        # 5MAが上→下に変化した区間（下向き区間の安値を探す）
        elif ma5_up[i - 1] and not ma5_up[i]:
            # 下向き区間の開始
            start = i
            while i < len(df) and not ma5_up[i]:
                i += 1
            end = i - 1
            # その区間の最低値（ヒゲ含む）
            segment = df.iloc[start:end + 1]
            if len(segment) > 0:
                min_idx = segment["Low"].idxmin()
                lows.append((min_idx, float(segment["Low"].min())))
        else:
            i += 1

    return highs, lows


# ────────────────────────────────────────────
# 支持線・抵抗線の特定
# ────────────────────────────────────────────

def find_support_resistance(
    df: pd.DataFrame,
    highs: list,
    lows: list,
    tolerance_pct: float = 0.015
) -> list[SupportResistanceLevel]:
    """
    よろち式高値安値から支持線・抵抗線を抽出し、
    反応回数・強度を算出する
    """
    levels = []

    # 高値から抵抗線を生成
    for date, price in highs:
        merged = False
        for lvl in levels:
            if lvl.level_type == "resistance" and abs(lvl.price - price) / price < tolerance_pct:
                # 近いラインと統合（平均化）
                lvl.price = (lvl.price * lvl.touch_count + price) / (lvl.touch_count + 1)
                lvl.touch_count += 1
                merged = True
                break
        if not merged:
            levels.append(SupportResistanceLevel(
                price=price, level_type="resistance",
                touch_count=1, source="高値安値"
            ))

    # 安値から支持線を生成
    for date, price in lows:
        merged = False
        for lvl in levels:
            if lvl.level_type == "support" and abs(lvl.price - price) / price < tolerance_pct:
                lvl.price = (lvl.price * lvl.touch_count + price) / (lvl.touch_count + 1)
                lvl.touch_count += 1
                merged = True
                break
        if not merged:
            levels.append(SupportResistanceLevel(
                price=price, level_type="support",
                touch_count=1, source="高値安値"
            ))

    # ギャップ由来のレベルを追加
    gaps = detect_gaps(df)
    for gap in gaps:
        levels.append(SupportResistanceLevel(
            price=gap.upper, level_type="resistance",
            touch_count=2, source="ギャップ"
        ))
        levels.append(SupportResistanceLevel(
            price=gap.lower, level_type="support",
            touch_count=2, source="ギャップ"
        ))

    # 毛抜き天井/底を検出して強化
    makenuki_levels = detect_makenuki(df)
    for price, lvl_type in makenuki_levels:
        for lvl in levels:
            if lvl.level_type == lvl_type and abs(lvl.price - price) / price < tolerance_pct:
                lvl.touch_count += 1
                lvl.source += "+毛抜き"
                break
        else:
            levels.append(SupportResistanceLevel(
                price=price, level_type=lvl_type,
                touch_count=2, source="毛抜き"
            ))

    # 強度ラベルを設定
    for lvl in levels:
        if lvl.touch_count >= 3:
            lvl.strength = "強"
        elif lvl.touch_count == 2:
            lvl.strength = "中"
        else:
            lvl.strength = "弱"

    # 現在価格に近いもの（±20%以内）に絞る
    current_price = float(df["Close"].iloc[-1])
    levels = [lvl for lvl in levels
              if abs(lvl.price - current_price) / current_price < 0.20]

    # 価格で整列
    levels.sort(key=lambda x: x.price, reverse=True)
    return levels


# ────────────────────────────────────────────
# ギャップ（窓）検出
# ────────────────────────────────────────────

def detect_gaps(df: pd.DataFrame) -> list[GapInfo]:
    """ギャップアップ/ダウンを検出"""
    gaps = []
    if len(df) < 2:
        return gaps

    for i in range(1, len(df)):
        prev_high = float(df.iloc[i - 1]["High"])
        prev_low = float(df.iloc[i - 1]["Low"])
        curr_high = float(df.iloc[i]["High"])
        curr_low = float(df.iloc[i]["Low"])
        date = df.index[i]

        if curr_low > prev_high:
            # ギャップアップ
            gap_pct = (curr_low - prev_high) / prev_high
            if gap_pct > 0.005:  # 0.5%以上のギャップ
                # 埋まったかチェック
                is_filled = any(
                    float(df.iloc[j]["Low"]) <= prev_high
                    for j in range(i + 1, min(i + 30, len(df)))
                )
                gaps.append(GapInfo(
                    date=date,
                    gap_type="up",
                    upper=curr_low,
                    lower=prev_high,
                    is_filled=is_filled
                ))

        elif curr_high < prev_low:
            # ギャップダウン
            gap_pct = (prev_low - curr_high) / prev_low
            if gap_pct > 0.005:
                is_filled = any(
                    float(df.iloc[j]["High"]) >= prev_low
                    for j in range(i + 1, min(i + 30, len(df)))
                )
                gaps.append(GapInfo(
                    date=date,
                    gap_type="down",
                    upper=prev_low,
                    lower=curr_high,
                    is_filled=is_filled
                ))

    # 直近20個に絞る
    return gaps[-20:]


# ────────────────────────────────────────────
# 毛抜き天井/底の検出
# ────────────────────────────────────────────

def detect_makenuki(df: pd.DataFrame, tolerance_pct: float = 0.005) -> list[tuple]:
    """
    毛抜き天井（複数足のヒゲ先端が同価格帯）
    毛抜き底（複数足の下ヒゲ先端が同価格帯）を検出

    Returns:
        [(price, level_type), ...]  level_type: "resistance" | "support"
    """
    results = []
    window = 20  # 過去20本で探す

    if len(df) < window:
        return results

    recent = df.iloc[-window:]

    # ヒゲ先端（高値）が近いもの
    highs = recent["High"].values
    for i in range(len(highs)):
        cluster = [highs[i]]
        for j in range(i + 1, len(highs)):
            if abs(highs[j] - highs[i]) / highs[i] < tolerance_pct:
                cluster.append(highs[j])
        if len(cluster) >= 2:
            avg_price = np.mean(cluster)
            results.append((avg_price, "resistance"))

    # ヒゲ先端（安値）が近いもの
    lows = recent["Low"].values
    for i in range(len(lows)):
        cluster = [lows[i]]
        for j in range(i + 1, len(lows)):
            if abs(lows[j] - lows[i]) / lows[i] < tolerance_pct:
                cluster.append(lows[j])
        if len(cluster) >= 2:
            avg_price = np.mean(cluster)
            results.append((avg_price, "support"))

    # 重複除去
    unique = []
    for price, lvl_type in results:
        if not any(abs(p - price) / price < tolerance_pct and t == lvl_type
                   for p, t in unique):
            unique.append((price, lvl_type))

    return unique


# ────────────────────────────────────────────
# 横ばい帯の検出
# ────────────────────────────────────────────

def detect_range_zones(df: pd.DataFrame, min_bars: int = 3, tolerance_pct: float = 0.02) -> list[dict]:
    """
    3本以上のローソク足が同価格帯に並ぶ横ばい帯を検出
    （よろち式：横ばい帯も支持抵抗として扱う）
    """
    zones = []
    if len(df) < min_bars:
        return zones

    i = 0
    while i < len(df) - min_bars + 1:
        base_high = float(df.iloc[i]["High"])
        base_low = float(df.iloc[i]["Low"])
        center = (base_high + base_low) / 2
        count = 1

        for j in range(i + 1, len(df)):
            h = float(df.iloc[j]["High"])
            lo = float(df.iloc[j]["Low"])
            c = (h + lo) / 2
            if abs(c - center) / center < tolerance_pct:
                count += 1
            else:
                break

        if count >= min_bars:
            zone_df = df.iloc[i:i + count]
            zones.append({
                "start": df.index[i],
                "end": df.index[i + count - 1],
                "upper": float(zone_df["High"].max()),
                "lower": float(zone_df["Low"].min()),
                "center": float(zone_df["Close"].mean()),
                "bar_count": count
            })
            i += count
        else:
            i += 1

    return zones


# ────────────────────────────────────────────
# カップ・ウィズ・ハンドル検出
# ────────────────────────────────────────────

def detect_cup_with_handle(df: pd.DataFrame) -> PatternResult:
    """
    カップ・ウィズ・ハンドルパターンを簡易検出
    （よろちさん：週足の買鉄形）
    """
    result = PatternResult(name="カップ・ウィズ・ハンドル", detected=False)
    if len(df) < 60:
        return result

    closes = df["Close"].values
    # 直近60〜120本で探す
    search_window = min(120, len(df))
    segment = closes[-search_window:]

    # 最高値の位置
    peak1_idx = np.argmax(segment[:len(segment) // 2])
    peak1 = segment[peak1_idx]

    # カップ底（peak1以降の最低値）
    cup_segment = segment[peak1_idx:]
    bottom_idx = np.argmin(cup_segment) + peak1_idx
    bottom = segment[bottom_idx]

    # カップの深さ（20〜33%が理想）
    depth = (peak1 - bottom) / peak1
    if not (0.12 < depth < 0.50):
        return result

    # 右側ピーク
    right_segment = segment[bottom_idx:]
    if len(right_segment) < 10:
        return result

    peak2_idx = np.argmax(right_segment) + bottom_idx
    peak2 = segment[peak2_idx]

    # 右側ピークがカップ左側ピークの90%以上
    if peak2 < peak1 * 0.85:
        return result

    # ハンドル（右ピーク後の小さな調整）
    handle_segment = segment[peak2_idx:]
    if len(handle_segment) < 5:
        return result

    handle_low = np.min(handle_segment)
    handle_depth = (peak2 - handle_low) / peak2

    if 0.02 < handle_depth < 0.15:
        result.detected = True
        result.confidence = min(0.9, 0.5 + (1 - abs(depth - 0.25)) + (1 - handle_depth))
        result.description = (
            f"カップ深さ: {depth:.1%} / ハンドル深さ: {handle_depth:.1%}\n"
            f"ブレイクアウト目標: {peak2 * 1.05:.2f}"
        )

    return result


# ────────────────────────────────────────────
# パターン分析（統合）
# ────────────────────────────────────────────

def analyze_patterns(df: pd.DataFrame) -> list[PatternResult]:
    """各種チャートパターンを検出"""
    patterns = []

    # カップ・ウィズ・ハンドル
    cwh = detect_cup_with_handle(df)
    patterns.append(cwh)

    # 三角収束（ペナント）
    tri = detect_triangle(df)
    patterns.append(tri)

    # ダブルボトム
    db = detect_double_bottom(df)
    patterns.append(db)

    return patterns


def detect_triangle(df: pd.DataFrame) -> PatternResult:
    """三角収束パターン（高値切り下がり＋安値切り上がり）"""
    result = PatternResult(name="三角収束（ペナント）", detected=False)
    if len(df) < 20:
        return result

    recent = df.iloc[-30:]
    highs = recent["High"].values
    lows = recent["Low"].values
    x = np.arange(len(highs))

    # 高値の回帰直線
    high_coef = np.polyfit(x, highs, 1)
    # 安値の回帰直線
    low_coef = np.polyfit(x, lows, 1)

    # 高値が切り下がり、安値が切り上がる
    if high_coef[0] < -0.01 and low_coef[0] > 0.01:
        result.detected = True
        result.confidence = 0.6
        result.description = "高値が切り下がり・安値が切り上がる収束形。ブレイク待ち。"

    return result


def detect_double_bottom(df: pd.DataFrame) -> PatternResult:
    """ダブルボトム（W底）検出"""
    result = PatternResult(name="ダブルボトム（W底）", detected=False,
                           description="※よろち式では「あまり強くない」パターン")
    if len(df) < 30:
        return result

    recent_lows = df["Low"].iloc[-40:].values
    x = np.arange(len(recent_lows))

    # 局所最小値を探す
    from scipy.signal import argrelextrema
    try:
        local_mins = argrelextrema(recent_lows, np.less, order=5)[0]
        if len(local_mins) >= 2:
            b1 = recent_lows[local_mins[-2]]
            b2 = recent_lows[local_mins[-1]]
            if abs(b1 - b2) / b1 < 0.03:  # 3%以内で同水準
                result.detected = True
                result.confidence = 0.5
                result.description += f"\n底値1: {b1:.2f} / 底値2: {b2:.2f}"
    except Exception:
        pass

    return result


# ────────────────────────────────────────────
# リスクリワード計算
# ────────────────────────────────────────────

def calc_risk_reward(
    entry: float,
    stop_loss: float,
    target: float
) -> dict:
    """
    リスクリワード比を計算する
    よろち式推奨: 1:5以上
    """
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)

    if risk == 0:
        return {"ratio": 0, "label": "計算不可", "ok": False}

    ratio = reward / risk

    return {
        "risk": risk,
        "reward": reward,
        "ratio": ratio,
        "ratio_label": f"1 : {ratio:.1f}",
        "ok": ratio >= 5.0,
        "comment": (
            "✅ よろち式推奨のリスクリワード（1:5以上）を満たしています"
            if ratio >= 5.0
            else f"⚠️ リスクリワードが低い（1:{ratio:.1f}）。よろち式推奨は1:5以上"
        )
    }


# ────────────────────────────────────────────
# 総合分析
# ────────────────────────────────────────────

def run_full_analysis(
    ticker_input: str,
    period: str = "1y",
    interval: str = "1d"
) -> Optional[AnalysisResult]:
    """
    全分析を実行してAnalysisResultを返す
    """
    df, symbol, company_name = fetch_stock_data(ticker_input, period, interval)
    if df is None or df.empty:
        return None

    trend = analyze_trend(df)
    kaitetsu = analyze_kaitetsu(df)
    highs, lows = extract_yorochi_highs_lows(df)
    sr_levels = find_support_resistance(df, highs, lows)
    gaps = detect_gaps(df)
    patterns = analyze_patterns(df)

    # 現在の5MA（損切り基準）
    current_ma5 = float(df["MA5"].iloc[-1]) if not pd.isna(df["MA5"].iloc[-1]) else 0
    current_price = float(df["Close"].iloc[-1])

    # 一般テクニカル指標（indicators.py）
    from indicators import run_all_indicators
    indicators = run_all_indicators(df)

    return AnalysisResult(
        ticker=symbol,
        df=df,
        trend=trend,
        kaitetsu=kaitetsu,
        support_resistance=sr_levels,
        gaps=gaps,
        patterns=patterns,
        high_low_marks={"highs": highs, "lows": lows},
        stop_loss_price=current_ma5,
        current_price=current_price,
        company_name=company_name,
        indicators=indicators,
    )


# ────────────────────────────────────────────
# 市場地合い取得
# ────────────────────────────────────────────

MARKET_WATCHLIST = {
    "ラッセル2000": "IWM",
    "VIX（恐怖指数）": "^VIX",
    "ビットコイン": "BTC-USD",
    "米国債10年利回り": "^TNX",
    "日経225": "^N225",
}


def get_market_overview() -> dict:
    """市場地合い指標を一括取得"""
    results = {}
    for name, symbol in MARKET_WATCHLIST.items():
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="5d", interval="1d")
            if not hist.empty and len(hist) >= 2:
                latest = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = (latest - prev) / prev * 100
                results[name] = {
                    "symbol": symbol,
                    "price": latest,
                    "change_pct": change_pct,
                    "up": change_pct >= 0
                }
        except Exception:
            pass
    return results
