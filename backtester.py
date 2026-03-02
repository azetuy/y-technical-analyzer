"""
バックテストエンジン

設計方針:
  - ルックアヘッドバイアスなし: シグナルは当日終値で判定、執行は翌日始値
  - ストップ確認: 当日のLowがストップ価格を下回ったら当日中に損切り
  - 手数料: 往復0.2%（現実的な設定、変更可）
  - ポジションサイズ: 資金の固定%（デフォルト10%）
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")


# ────────────────────────────────────────────
# データ構造
# ────────────────────────────────────────────

@dataclass
class Trade:
    trade_id: int
    ticker: str
    strategy_name: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop_price: float = 0.0          # 損切り価格
    target_price: float = 0.0        # 目標価格
    entry_signal_detail: str = ""    # シグナル内容メモ
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = "保有中"      # "損切り" | "利確" | "タイムストップ" | "シグナル反転"

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None

    @property
    def pnl(self) -> float:
        if not self.is_closed:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if not self.is_closed or self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def holding_days(self) -> int:
        if not self.is_closed:
            return 0
        return (self.exit_date - self.entry_date).days

    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0

    @property
    def r_multiple(self) -> float:
        """Rマルチプル（リスク1に対するリワード）"""
        if self.entry_price == 0 or self.stop_price == 0:
            return 0.0
        risk_pct = abs(self.entry_price - self.stop_price) / self.entry_price
        return self.pnl_pct / 100 / risk_pct if risk_pct > 0 else 0.0


@dataclass
class BacktestResult:
    ticker: str
    strategy_name: str
    period: str
    initial_capital: float
    commission_rate: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)

    @property
    def final_capital(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) > 0 else self.initial_capital

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.is_closed]

    @property
    def total_return_pct(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    @property
    def annual_return_pct(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days == 0:
            return 0.0
        return ((self.final_capital / self.initial_capital) ** (365 / days) - 1) * 100

    @property
    def win_rate(self) -> float:
        ct = self.closed_trades
        if not ct:
            return 0.0
        return len([t for t in ct if t.is_win]) / len(ct) * 100

    @property
    def profit_factor(self) -> float:
        ct = self.closed_trades
        gross_profit = sum(t.pnl for t in ct if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in ct if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown_pct(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100
        return float(drawdown.min())

    @property
    def sharpe_ratio(self) -> float:
        r = self.daily_returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))

    @property
    def sortino_ratio(self) -> float:
        r = self.daily_returns.dropna()
        downside = r[r < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return float(r.mean() / downside.std() * np.sqrt(252))

    @property
    def avg_win_pct(self) -> float:
        wins = [t.pnl_pct for t in self.closed_trades if t.is_win]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pnl_pct for t in self.closed_trades if not t.is_win]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def avg_holding_days(self) -> float:
        ct = self.closed_trades
        return float(np.mean([t.holding_days for t in ct])) if ct else 0.0

    @property
    def avg_r_multiple(self) -> float:
        ct = self.closed_trades
        return float(np.mean([t.r_multiple for t in ct])) if ct else 0.0

    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def exit_reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.closed_trades:
            counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
        return counts

    def monthly_returns(self) -> pd.DataFrame:
        """月次リターンのピボットテーブル（年×月）"""
        if len(self.equity_curve) < 2:
            return pd.DataFrame()
        monthly = self.equity_curve.resample("ME").last().pct_change() * 100
        monthly.index = monthly.index.to_period("M")
        df = monthly.reset_index()
        df.columns = ["period", "return"]
        df["year"] = df["period"].dt.year
        df["month"] = df["period"].dt.month
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [f"{m}月" for m in pivot.columns]
        return pivot


# ────────────────────────────────────────────
# シグナル計算（ルックアヘッドなし）
# ────────────────────────────────────────────

def _calc_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    全シグナルを前処理として計算する
    pandas の rolling/ewm は backward-looking なのでルックアヘッドバイアスなし
    """
    sig = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── 移動平均線 ─────────────────────────────
    sig["ma5"] = close.rolling(5).mean()
    sig["ma20"] = close.rolling(20).mean()
    sig["ma60"] = close.rolling(60).mean()
    sig["ma5_slope"] = sig["ma5"].diff(3)
    sig["ma20_slope"] = sig["ma20"].diff(5)
    sig["ma5_up"] = sig["ma5_slope"] > 0

    # ── 買鉄5条件（よろち式） ─────────────────
    sig["k1"] = sig["ma5"] > sig["ma20"]                          # 5MA > 20MA
    sig["k2"] = sig["ma20_slope"] > 0                              # 20MA上向き
    sig["k3"] = (~sig["ma5_up"].shift(1).fillna(False)) & sig["ma5_up"]  # 5MA転換
    body_bottom = close.combine(df["Open"], min)
    body_top = close.combine(df["Open"], max)
    body_len = (body_top - body_bottom).replace(0, np.nan)
    above_ma5 = (body_top - sig["ma5"]).clip(lower=0)
    sig["k4"] = (above_ma5 / body_len >= 0.5) & (close > sig["ma5"])
    # K5: 直近10本でMA20に3%以内接近
    def k5_check(i):
        if i < 10:
            return False
        for j in range(max(0, i - 10), i):
            ma20_j = sig["ma20"].iloc[j]
            low_j = low.iloc[j]
            if ma20_j > 0 and abs(low_j - ma20_j) / ma20_j < 0.03:
                return True
        return False
    sig["k5"] = pd.Series(
        [k5_check(i) for i in range(len(sig))],
        index=sig.index
    )
    sig["kaitetsu_score"] = sig[["k1", "k2", "k3", "k4", "k5"]].sum(axis=1)

    # ── RSI(14) ─────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    sig["rsi"] = 100 - (100 / (1 + rs))
    sig["rsi_oversold"] = sig["rsi"] < 30
    sig["rsi_overbought"] = sig["rsi"] > 70

    # ── MACD(12,26,9) ────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    sig["macd"] = ema12 - ema26
    sig["macd_signal"] = sig["macd"].ewm(span=9, adjust=False).mean()
    sig["macd_hist"] = sig["macd"] - sig["macd_signal"]
    sig["macd_golden"] = (sig["macd"].shift(1) < sig["macd_signal"].shift(1)) & (sig["macd"] > sig["macd_signal"])
    sig["macd_dead"] = (sig["macd"].shift(1) > sig["macd_signal"].shift(1)) & (sig["macd"] < sig["macd_signal"])
    sig["macd_above"] = sig["macd"] > sig["macd_signal"]

    # ── Bollinger Bands(20,2σ) ───────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    sig["bb_upper"] = bb_mid + 2 * bb_std
    sig["bb_lower"] = bb_mid - 2 * bb_std
    bb_range = (sig["bb_upper"] - sig["bb_lower"]).replace(0, np.nan)
    sig["bb_pct_b"] = (close - sig["bb_lower"]) / bb_range
    sig["bb_near_lower"] = sig["bb_pct_b"] < 0.1

    # ── OBV ──────────────────────────────────
    direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
    sig["obv"] = (direction * volume).cumsum()
    sig["obv_slope"] = sig["obv"].diff(10)

    # ── 一目均衡表 ───────────────────────────
    def midpoint(h, lo, n):
        return (h.rolling(n).max() + lo.rolling(n).min()) / 2
    sig["tenkan"] = midpoint(high, low, 9)
    sig["kijun"] = midpoint(high, low, 26)
    senkou_a = ((sig["tenkan"] + sig["tenkan"]) / 2).shift(26)
    senkou_b = midpoint(high, low, 52).shift(26)
    sig["above_cloud"] = close > senkou_a.combine(senkou_b, max)

    # ── 複合スコア（簡易版）─────────────────
    score = pd.Series(0.0, index=sig.index)
    # RSI
    score += np.where(sig["rsi"] <= 25, 2.0, np.where(sig["rsi"] <= 35, 1.0,
             np.where(sig["rsi"] >= 75, -2.0, np.where(sig["rsi"] >= 65, -1.0, 0.0))))
    # MACD
    score += np.where(sig["macd_golden"], 2.0, np.where(sig["macd_dead"], -2.0,
             np.where(sig["macd_above"], 0.5, -0.5)))
    # BB
    score += np.where(sig["bb_pct_b"] < 0.1, 1.0, np.where(sig["bb_pct_b"] > 0.9, -0.5, 0.0))
    # Ichimoku
    score += np.where(sig["above_cloud"], 1.5, -1.5)
    # OBV trend
    score += np.where(sig["obv_slope"] > 0, 0.5, -0.5)
    # MA trend
    ma5_up_num = sig["ma5_up"].astype(float)
    ma20_up_num = (sig["ma20_slope"] > 0).astype(float)
    score += (ma5_up_num + ma20_up_num) * 0.5
    sig["composite_score"] = score.clip(-10, 10)

    return sig


def _calc_resistance_target(df: pd.DataFrame, entry_idx: int, lookback: int = 60) -> float:
    """
    エントリー時点から過去データを使って直近の抵抗線を算出
    ルックアヘッドバイアスなし（過去データのみ使用）
    """
    start = max(0, entry_idx - lookback)
    recent = df.iloc[start:entry_idx]
    if recent.empty:
        return 0.0
    current_close = float(df["Close"].iloc[entry_idx])
    # 現在値より上の高値を探す
    highs_above = recent["High"][recent["High"] > current_close]
    if highs_above.empty:
        return current_close * 1.10  # デフォルト: 10%上
    return float(highs_above.min())  # 最も近い抵抗


# ────────────────────────────────────────────
# バックテストエンジン
# ────────────────────────────────────────────

class BacktestEngine:
    """
    単一銘柄のバックテストエンジン

    ルールまとめ:
    - シグナル判定: 当日終値
    - 執行: 翌日始値（スリッページ込みで±0.05%）
    - ストップ確認: 当日Low < ストップ価格 → ストップ価格で執行
    - 手数料: 往復 commission_rate × 2
    - ポジション: 同時1ポジションのみ（オーバーラップなし）
    """

    def __init__(
        self,
        commission_rate: float = 0.001,   # 片道0.1%
        slippage_rate: float = 0.0005,    # 片道0.05%
        position_size_pct: float = 10.0,  # 資金の10%
        max_holding_days: int = 30,       # タイムストップ（営業日ベース）
    ):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_size_pct = position_size_pct
        self.max_holding_days = max_holding_days

    def _apply_entry_cost(self, price: float) -> float:
        """エントリー時の手数料+スリッページを適用した実効コスト"""
        return price * (1 + self.commission_rate + self.slippage_rate)

    def _apply_exit_revenue(self, price: float) -> float:
        """エグジット時の手数料+スリッページを適用した実効収入"""
        return price * (1 - self.commission_rate - self.slippage_rate)

    def run(
        self,
        ticker_input: str,
        strategy_name: str,
        strategy_cfg: dict,
        period: str = "5y",
        initial_capital: float = 1_000_000,
    ) -> BacktestResult:
        """
        バックテストを実行する

        Parameters:
            ticker_input: ティッカー or 証券コード
            strategy_name: 戦略名
            strategy_cfg: 戦略設定dict (strategies.py参照)
            period: 期間 ("1y", "2y", "5y", etc.)
            initial_capital: 初期資金（円 or ドル）
        """
        from analyzer import get_ticker_symbol
        symbol = get_ticker_symbol(ticker_input)

        # データ取得
        tk = yf.Ticker(symbol)
        df = tk.history(period=period, interval="1d", auto_adjust=True)
        if df.empty or len(df) < 60:
            raise ValueError(f"データ不足: {symbol}")

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # シグナル計算
        sig = _calc_all_signals(df)

        # バックテスト実行
        capital = initial_capital
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        equity_values = []
        trade_id = 0

        for i in range(1, len(df) - 1):  # -1: 翌日始値が必要
            date = df.index[i]
            o = float(df["Open"].iloc[i])
            h = float(df["High"].iloc[i])
            lo = float(df["Low"].iloc[i])
            c = float(df["Close"].iloc[i])

            # ─── オープンポジションの確認 ───────
            if open_trade is not None:
                exited = False

                # 1. ストップロス確認（当日Low < ストップ価格）
                if lo <= open_trade.stop_price and open_trade.stop_price > 0:
                    exit_price = max(lo, open_trade.stop_price)  # ストップ価格で執行
                    if o < open_trade.stop_price:
                        exit_price = o  # 窓開け下落の場合は始値
                    revenue = self._apply_exit_revenue(exit_price)
                    capital += revenue * open_trade.shares
                    open_trade.exit_date = date
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = "損切り"
                    trades.append(open_trade)
                    open_trade = None
                    exited = True

                # 2. 利確目標確認（当日High > 目標価格）
                if not exited and open_trade and open_trade.target_price > 0 and h >= open_trade.target_price:
                    exit_price = open_trade.target_price
                    revenue = self._apply_exit_revenue(exit_price)
                    capital += revenue * open_trade.shares
                    open_trade.exit_date = date
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = "利確"
                    trades.append(open_trade)
                    open_trade = None
                    exited = True

                # 3. タイムストップ（保有日数超過）
                if not exited and open_trade:
                    trading_days = sum(1 for j in range(len(df))
                                      if df.index[j] > open_trade.entry_date and df.index[j] <= date)
                    if trading_days >= self.max_holding_days:
                        revenue = self._apply_exit_revenue(o)  # 翌日始値で出る
                        capital += revenue * open_trade.shares
                        open_trade.exit_date = date
                        open_trade.exit_price = o
                        open_trade.exit_reason = "タイムストップ"
                        trades.append(open_trade)
                        open_trade = None
                        exited = True

                # 4. ストラテジー固有の出口シグナル（当日終値で判定）
                if not exited and open_trade:
                    exit_signal, exit_reason = self._check_exit_signal(
                        sig, i, open_trade, strategy_cfg, c
                    )
                    if exit_signal:
                        # 次の始値で出る（翌日）
                        next_open = float(df["Open"].iloc[i + 1]) if i + 1 < len(df) else c
                        revenue = self._apply_exit_revenue(next_open)
                        capital += revenue * open_trade.shares
                        open_trade.exit_date = df.index[i + 1] if i + 1 < len(df) else date
                        open_trade.exit_price = next_open
                        open_trade.exit_reason = exit_reason
                        trades.append(open_trade)
                        open_trade = None

            # ─── エントリーシグナル確認 ───────────
            if open_trade is None:
                entry_signal, signal_detail = self._check_entry_signal(sig, i, strategy_cfg)

                if entry_signal:
                    # 翌日始値でエントリー
                    next_open = float(df["Open"].iloc[i + 1])
                    entry_cost = self._apply_entry_cost(next_open)
                    position_value = capital * (self.position_size_pct / 100)
                    shares = position_value / entry_cost

                    # ストップ価格・目標価格の設定
                    stop_price = self._calc_stop(df, sig, i, next_open, strategy_cfg)
                    target_price = self._calc_target(df, i + 1, next_open, strategy_cfg)

                    if shares > 0:
                        capital -= entry_cost * shares
                        trade_id += 1
                        open_trade = Trade(
                            trade_id=trade_id,
                            ticker=symbol,
                            strategy_name=strategy_name,
                            entry_date=df.index[i + 1],
                            entry_price=next_open,
                            shares=shares,
                            stop_price=stop_price,
                            target_price=target_price,
                            entry_signal_detail=signal_detail,
                        )

            # ─── エクイティカーブ更新 ────────────
            position_value = 0.0
            if open_trade is not None:
                position_value = c * open_trade.shares
            equity_values.append(capital + position_value)

        # 未決済ポジションの強制決済
        if open_trade is not None:
            last_close = float(df["Close"].iloc[-1])
            revenue = self._apply_exit_revenue(last_close)
            capital += revenue * open_trade.shares
            open_trade.exit_date = df.index[-1]
            open_trade.exit_price = last_close
            open_trade.exit_reason = "期間終了"
            trades.append(open_trade)
            equity_values[-1] = capital

        equity_series = pd.Series(equity_values, index=df.index[1:len(equity_values) + 1])
        daily_returns = equity_series.pct_change().dropna()

        company_name = tk.info.get("longName", ticker_input)
        result = BacktestResult(
            ticker=f"{symbol}（{company_name}）",
            strategy_name=strategy_name,
            period=period,
            initial_capital=initial_capital,
            commission_rate=self.commission_rate,
            trades=trades,
            equity_curve=equity_series,
            daily_returns=daily_returns,
        )
        return result

    def _check_entry_signal(self, sig: pd.DataFrame, i: int, cfg: dict) -> tuple[bool, str]:
        """エントリーシグナルを判定する（当日終値ベース）"""
        condition = cfg.get("entry_condition", "")

        if condition == "kaitetsu":
            threshold = cfg.get("entry_threshold", 5)
            score = int(sig["kaitetsu_score"].iloc[i])
            if score >= threshold:
                return True, f"買鉄スコア {score}/5"

        elif condition == "rsi_oversold":
            rsi = sig["rsi"].iloc[i]
            threshold = cfg.get("entry_threshold", 30)
            # RSI が閾値を下から上に抜けた（確認された反転）
            prev_rsi = sig["rsi"].iloc[i - 1] if i > 0 else rsi
            ma20_up = sig["ma20_slope"].iloc[i] > 0
            if prev_rsi < threshold and rsi > threshold and ma20_up:
                return True, f"RSI反転 {prev_rsi:.0f}→{rsi:.0f}"

        elif condition == "macd_golden":
            if sig["macd_golden"].iloc[i]:
                macd_val = sig["macd"].iloc[i]
                return True, f"MACDゴールデンクロス {macd_val:.3f}"

        elif condition == "composite":
            threshold = cfg.get("entry_threshold", 3.0)
            score = sig["composite_score"].iloc[i]
            prev_score = sig["composite_score"].iloc[i - 1] if i > 0 else score
            # スコアが閾値を超えてきた
            if prev_score < threshold and score >= threshold:
                return True, f"複合スコア {prev_score:.1f}→{score:.1f}"

        elif condition == "kaitetsu_plus_composite":
            k_min = cfg.get("kaitetsu_min", 4)
            c_min = cfg.get("composite_min", 2.0)
            k_score = int(sig["kaitetsu_score"].iloc[i])
            c_score = sig["composite_score"].iloc[i]
            if k_score >= k_min and c_score >= c_min:
                return True, f"買鉄{k_score}/5 + 複合{c_score:.1f}"

        elif condition == "bb_squeeze_break":
            # BBスクイーズ後のブレイクアウト
            bb_pct = sig["bb_pct_b"].iloc[i]
            prev_pct = sig["bb_pct_b"].iloc[i - 1] if i > 0 else bb_pct
            if prev_pct < 0.2 and bb_pct > 0.5:  # 下限付近から中央を超えた
                ma_up = sig["ma5_up"].iloc[i]
                if ma_up:
                    return True, f"BBブレイクアウト %B={bb_pct:.2f}"

        return False, ""

    def _check_exit_signal(
        self, sig: pd.DataFrame, i: int, trade: Trade, cfg: dict, current_close: float
    ) -> tuple[bool, str]:
        """出口シグナルを判定する（当日終値ベース）"""
        exit_stop = cfg.get("exit_stop", "ma5_break")

        if exit_stop == "ma5_break":
            ma5 = sig["ma5"].iloc[i]
            if not pd.isna(ma5) and current_close < ma5:
                return True, "シグナル反転（5MA割れ）"

        elif exit_stop == "macd_dead":
            if sig["macd_dead"].iloc[i]:
                return True, "MACDデッドクロス"

        elif exit_stop == "composite_negative":
            threshold = cfg.get("exit_stop_threshold", -1.0)
            score = sig["composite_score"].iloc[i]
            if score < threshold:
                return True, f"スコア悪化（{score:.1f}）"

        elif exit_stop == "rsi_overbought":
            threshold = cfg.get("exit_target_threshold", 65)
            if sig["rsi"].iloc[i] > threshold:
                return True, f"RSI買われすぎ（{sig['rsi'].iloc[i]:.0f}）"

        return False, ""

    def _calc_stop(
        self, df: pd.DataFrame, sig: pd.DataFrame, signal_idx: int,
        entry_price: float, cfg: dict
    ) -> float:
        """損切り価格を計算"""
        stop_type = cfg.get("stop_type", "ma5")

        if stop_type == "ma5":
            ma5 = float(sig["ma5"].iloc[signal_idx])
            return ma5 if ma5 > 0 else entry_price * 0.95

        elif stop_type == "fixed_pct":
            pct = cfg.get("stop_pct", 5.0) / 100
            return entry_price * (1 - pct)

        elif stop_type == "atr":
            # ATR × 2
            hi = df["High"]
            lo = df["Low"]
            pc = df["Close"].shift(1)
            tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
            atr = float(tr.ewm(span=14, adjust=False).mean().iloc[signal_idx])
            return entry_price - atr * 2

        return entry_price * 0.95

    def _calc_target(
        self, df: pd.DataFrame, entry_idx: int,
        entry_price: float, cfg: dict
    ) -> float:
        """目標価格を計算"""
        target_type = cfg.get("target_type", "resistance")

        if target_type == "resistance":
            target = _calc_resistance_target(df, entry_idx)
            # 目標が現在値の5%以上上でなければデフォルト
            if target < entry_price * 1.05:
                return entry_price * 1.10
            return target

        elif target_type == "rr_ratio":
            rr = cfg.get("target_rr", 3.0)
            stop = self._calc_stop(df, pd.DataFrame(), entry_idx, entry_price, cfg)
            risk = entry_price - stop
            return entry_price + risk * rr

        return 0.0  # 目標なし（タイムストップのみ）


# ────────────────────────────────────────────
# 複数銘柄バッチ実行
# ────────────────────────────────────────────

def run_batch_backtest(
    tickers: List[str],
    strategy_name: str,
    strategy_cfg: dict,
    period: str = "3y",
    initial_capital: float = 1_000_000,
    **engine_kwargs,
) -> Dict[str, BacktestResult]:
    """
    複数銘柄に対して同じ戦略でバックテストを実行する
    """
    engine = BacktestEngine(**engine_kwargs)
    results = {}

    for ticker in tickers:
        try:
            result = engine.run(ticker, strategy_name, strategy_cfg, period, initial_capital)
            results[ticker] = result
        except Exception as e:
            print(f"スキップ [{ticker}]: {e}")

    return results


def summarize_batch(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """バッチ結果を比較可能なサマリーDataFrameに変換"""
    rows = []
    for ticker, r in results.items():
        rows.append({
            "銘柄": ticker,
            "総リターン(%)": round(r.total_return_pct, 2),
            "年率リターン(%)": round(r.annual_return_pct, 2),
            "勝率(%)": round(r.win_rate, 1),
            "プロフィットファクター": round(r.profit_factor, 2) if r.profit_factor != float("inf") else "∞",
            "最大DD(%)": round(r.max_drawdown_pct, 2),
            "シャープ比": round(r.sharpe_ratio, 2),
            "総トレード数": r.total_trades,
            "平均保有日数": round(r.avg_holding_days, 1),
            "平均勝ち(%)": round(r.avg_win_pct, 2),
            "平均負け(%)": round(r.avg_loss_pct, 2),
        })
    return pd.DataFrame(rows).sort_values("総リターン(%)", ascending=False)
