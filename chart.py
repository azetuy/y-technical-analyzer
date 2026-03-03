"""
Plotlyチャート生成モジュール
Y式テクニカル分析ツール用
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analyzer import AnalysisResult, SupportResistanceLevel, GapInfo

# Bollinger Band表示用にメインチャートへ追加
COLORS_BB = {
    "bb_upper": "rgba(100,180,255,0.6)",
    "bb_mid": "rgba(100,180,255,0.4)",
    "bb_lower": "rgba(100,180,255,0.6)",
    "bb_fill": "rgba(100,180,255,0.05)",
    "ichimoku_up": "rgba(0,200,100,0.15)",
    "ichimoku_down": "rgba(255,80,80,0.15)",
}


# ────────────────────────────────────────────
# カラー定義
# ────────────────────────────────────────────

COLORS = {
    "candle_up": "#26A69A",        # 陽線（緑系）
    "candle_down": "#EF5350",      # 陰線（赤系）
    "ma5": "#FFA500",              # 5MA: オレンジ
    "ma20": "#00BFFF",             # 20MA: 水色
    "ma60": "#FF69B4",             # 60MA: ピンク
    "ma400": "#9370DB",            # 400MA: 紫（20ヶ月線）
    "resistance_strong": "#FF3333",  # 強い抵抗線
    "resistance_weak": "#FF9999",    # 弱い抵抗線
    "support_strong": "#3366FF",   # 強い支持線
    "support_weak": "#9999FF",     # 弱い支持線
    "high_mark": "#FF6600",        # Y式高値マーク
    "low_mark": "#0066FF",         # Y式安値マーク
    "kaitetsu": "#00FF88",         # 買鉄シグナル
    "gap_up": "rgba(0,200,100,0.1)",   # ギャップアップ背景
    "gap_down": "rgba(255,50,50,0.1)",  # ギャップダウン背景
    "volume_up": "rgba(38,166,154,0.5)",
    "volume_down": "rgba(239,83,80,0.5)",
    "bg": "#0E1117",               # 背景
    "grid": "#1E2A35",             # グリッド線
    "text": "#FFFFFF",             # テキスト
}


# ────────────────────────────────────────────
# メインチャート生成
# ────────────────────────────────────────────

def create_main_chart(result: AnalysisResult, show_sr: bool = True,
                      show_gaps: bool = True, show_highs_lows: bool = True,
                      show_kaitetsu: bool = True,
                      show_bb: bool = True, show_ichimoku: bool = False) -> go.Figure:
    """
    メインのローソク足チャートを生成する

    Parameters:
        result: 分析結果
        show_sr: 支持線・抵抗線を表示するか
        show_gaps: ギャップを表示するか
        show_highs_lows: Y式高値安値マークを表示するか
        show_kaitetsu: 買鉄シグナルを表示するか
    """
    df = result.df
    long_term_label = getattr(result, "long_term_ma_label", "400日線")

    # サブプロット（メイン + 出来高）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
        subplot_titles=["", "出来高"]
    )

    # ── ローソク足 ──────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="ローソク足",
            increasing=dict(line=dict(color=COLORS["candle_up"]), fillcolor=COLORS["candle_up"]),
            decreasing=dict(line=dict(color=COLORS["candle_down"]), fillcolor=COLORS["candle_down"]),
            hoverinfo="x+y",
        ),
        row=1, col=1
    )

    # ── 移動平均線 ──────────────────────────
    ma_configs = [
        ("MA5",   "5MA",   COLORS["ma5"],   1.5),
        ("MA20",  "20MA",  COLORS["ma20"],  2.0),
        ("MA60",  "60MA",  COLORS["ma60"],  1.5),
        ("MA400", long_term_label, COLORS["ma400"], 2.5),
    ]

    for col_name, label, color, width in ma_configs:
        if col_name in df.columns and not df[col_name].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    name=label,
                    line=dict(color=color, width=width),
                    mode="lines",
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                ),
                row=1, col=1
            )

    # ── ギャップ（窓）ハイライト ─────────────
    if show_gaps:
        for gap in result.gaps:
            if not gap.is_filled:
                color = COLORS["gap_up"] if gap.gap_type == "up" else COLORS["gap_down"]
                fig.add_hrect(
                    y0=gap.lower, y1=gap.upper,
                    fillcolor=color,
                    line_width=0,
                    row=1, col=1
                )
                # ギャップライン
                line_color = "#00CC66" if gap.gap_type == "up" else "#FF4444"
                fig.add_hline(
                    y=gap.upper, line_dash="dot",
                    line_color=line_color, line_width=1,
                    annotation_text=f"窓{'上' if gap.gap_type == 'up' else '下'}",
                    annotation_position="right",
                    row=1, col=1
                )
                fig.add_hline(
                    y=gap.lower, line_dash="dot",
                    line_color=line_color, line_width=1,
                    row=1, col=1
                )

    # ── 支持線・抵抗線 ──────────────────────
    if show_sr:
        _add_support_resistance(fig, result.support_resistance, df)

    # ── Y式 高値安値マーク ─────────────
    if show_highs_lows:
        _add_high_low_marks(fig, result.high_low_marks)

    # ── 買鉄シグナルマーク ──────────────────
    if show_kaitetsu and result.kaitetsu.is_valid:
        last_date = df.index[-1]
        last_close = float(df["Close"].iloc[-1])
        ma5_val = float(df["MA5"].iloc[-1]) if not pd.isna(df["MA5"].iloc[-1]) else last_close
        fig.add_trace(
            go.Scatter(
                x=[last_date],
                y=[min(float(df["Low"].iloc[-1]), ma5_val) * 0.99],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=20,
                    color=COLORS["kaitetsu"],
                    line=dict(width=1, color="white")
                ),
                text=["買鉄↑"],
                textposition="bottom center",
                textfont=dict(color=COLORS["kaitetsu"], size=11),
                name="買鉄シグナル",
                hovertemplate="買鉄シグナル<br>%{x}<extra></extra>",
            ),
            row=1, col=1
        )

    # ── 5MA方向転換マーク（Y式高値安値の起点）──
    _add_ma5_direction_changes(fig, df)

    # ── ボリンジャーバンド ──────────────────
    if show_bb and result.indicators:
        _add_bollinger_bands(fig, result.indicators, df)

    # ── 一目均衡表 ──────────────────────────
    if show_ichimoku and result.indicators:
        _add_ichimoku(fig, result.indicators, df)

    # ── 出来高 ──────────────────────────────
    colors_vol = [
        COLORS["volume_up"] if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
        else COLORS["volume_down"]
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="出来高",
            marker_color=colors_vol,
            showlegend=False,
            hovertemplate="出来高: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1
    )

    # ── レイアウト設定 ─────────────────────
    ticker_display = result.ticker.replace(".T", "（日本株）")
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker_display}</b>　{result.company_name}",
            font=dict(size=18, color=COLORS["text"]),
            x=0.01
        ),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Noto Sans JP, sans-serif"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode="x unified",
        height=700,
        margin=dict(l=10, r=60, t=80, b=10),
    )

    # グリッド設定
    for row in [1, 2]:
        fig.update_xaxes(
            gridcolor=COLORS["grid"], gridwidth=1,
            showline=True, linecolor="gray",
            row=row, col=1
        )
        fig.update_yaxes(
            gridcolor=COLORS["grid"], gridwidth=1,
            showline=True, linecolor="gray",
            tickformat=".2f",
            side="right",
            row=row, col=1
        )

    return fig


# ────────────────────────────────────────────
# 支持線・抵抗線の追加
# ────────────────────────────────────────────

def _add_support_resistance(
    fig: go.Figure,
    levels: list[SupportResistanceLevel],
    df: pd.DataFrame
) -> None:
    """支持線・抵抗線をチャートに追加"""
    current_price = float(df["Close"].iloc[-1])

    added_resistance = False
    added_support = False

    for lvl in levels:
        if lvl.level_type == "resistance":
            color = COLORS["resistance_strong"] if lvl.touch_count >= 3 else COLORS["resistance_weak"]
            dash = "solid" if lvl.touch_count >= 3 else "dash"
            label = f"抵抗 {lvl.price:.2f}（{lvl.touch_count}回 {lvl.source}）"
            show_legend = not added_resistance
            added_resistance = True
        else:
            color = COLORS["support_strong"] if lvl.touch_count >= 3 else COLORS["support_weak"]
            dash = "solid" if lvl.touch_count >= 3 else "dash"
            label = f"支持 {lvl.price:.2f}（{lvl.touch_count}回 {lvl.source}）"
            show_legend = not added_support
            added_support = True

        width = lvl.width
        dist_pct = (lvl.price - current_price) / current_price * 100
        annotation = f"{'▲' if lvl.level_type == 'resistance' else '▼'}{lvl.price:.2f} ({dist_pct:+.1f}%)"

        fig.add_hline(
            y=lvl.price,
            line_dash=dash,
            line_color=color,
            line_width=width,
            annotation_text=annotation,
            annotation_position="right",
            annotation_font=dict(color=color, size=10),
            row=1, col=1
        )


# ────────────────────────────────────────────
# Y式高値安値マークの追加
# ────────────────────────────────────────────

def _add_high_low_marks(fig: go.Figure, high_low_marks: dict) -> None:
    """Y式の高値・安値をチャートにマーク"""
    highs = high_low_marks.get("highs", [])
    lows = high_low_marks.get("lows", [])

    # 直近15個のみ表示
    highs_display = highs[-15:] if highs else []
    lows_display = lows[-15:] if lows else []

    if highs_display:
        dates_h, prices_h = zip(*highs_display)
        fig.add_trace(
            go.Scatter(
                x=list(dates_h),
                y=[p * 1.005 for p in prices_h],  # 少し上にオフセット
                mode="markers+text",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color=COLORS["high_mark"],
                    line=dict(width=1, color="white")
                ),
                text=["H" for _ in dates_h],
                textposition="top center",
                textfont=dict(color=COLORS["high_mark"], size=9),
                name="Y式高値",
                hovertemplate="Y式高値: %{y:.2f}<br>%{x}<extra></extra>",
            ),
            row=1, col=1
        )

    if lows_display:
        dates_l, prices_l = zip(*lows_display)
        fig.add_trace(
            go.Scatter(
                x=list(dates_l),
                y=[p * 0.995 for p in prices_l],  # 少し下にオフセット
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color=COLORS["low_mark"],
                    line=dict(width=1, color="white")
                ),
                text=["L" for _ in dates_l],
                textposition="bottom center",
                textfont=dict(color=COLORS["low_mark"], size=9),
                name="Y式安値",
                hovertemplate="Y式安値: %{y:.2f}<br>%{x}<extra></extra>",
            ),
            row=1, col=1
        )


# ────────────────────────────────────────────
# 5MA方向転換マーク
# ────────────────────────────────────────────

def _add_ma5_direction_changes(fig: go.Figure, df: pd.DataFrame) -> None:
    """5MAの方向転換点（Y式高値安値の区切り）をマーク"""
    if "MA5_up" not in df.columns:
        return

    ma5_up = df["MA5_up"].values
    turn_up_dates = []    # 下→上（安値確定）
    turn_down_dates = []  # 上→下（高値確定）

    for i in range(1, len(df)):
        if not ma5_up[i - 1] and ma5_up[i]:
            turn_up_dates.append((df.index[i], float(df["Low"].iloc[i])))
        elif ma5_up[i - 1] and not ma5_up[i]:
            turn_down_dates.append((df.index[i], float(df["High"].iloc[i])))

    # 直近10個
    turn_up_dates = turn_up_dates[-10:]
    turn_down_dates = turn_down_dates[-10:]

    if turn_up_dates:
        dates, prices = zip(*turn_up_dates)
        fig.add_trace(
            go.Scatter(
                x=list(dates),
                y=[p * 0.992 for p in prices],
                mode="markers",
                marker=dict(symbol="circle", size=7, color="#00FFCC",
                            line=dict(width=1, color="white")),
                name="5MA転換↑",
                hovertemplate="5MA上転換<br>%{x}<extra></extra>",
            ),
            row=1, col=1
        )

    if turn_down_dates:
        dates, prices = zip(*turn_down_dates)
        fig.add_trace(
            go.Scatter(
                x=list(dates),
                y=[p * 1.008 for p in prices],
                mode="markers",
                marker=dict(symbol="circle", size=7, color="#FF8C00",
                            line=dict(width=1, color="white")),
                name="5MA転換↓",
                hovertemplate="5MA下転換<br>%{x}<extra></extra>",
            ),
            row=1, col=1
        )


# ────────────────────────────────────────────
# ゲージ（リスクリワード）チャート
# ────────────────────────────────────────────

def create_rr_gauge(ratio: float) -> go.Figure:
    """リスクリワード比のゲージを生成"""
    max_val = max(10, ratio * 1.2)
    color = "#00CC66" if ratio >= 5 else "#FF6600" if ratio >= 3 else "#FF3333"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        title={"text": "リスクリワード比", "font": {"size": 14, "color": "white"}},
        delta={"reference": 5, "increasing": {"color": "#00CC66"},
               "decreasing": {"color": "#FF3333"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "white",
                     "tickfont": {"color": "white"}},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 3], "color": "rgba(255,50,50,0.3)"},
                {"range": [3, 5], "color": "rgba(255,165,0,0.3)"},
                {"range": [5, max_val], "color": "rgba(0,200,100,0.3)"},
            ],
            "threshold": {
                "line": {"color": "#FFD700", "width": 3},
                "thickness": 0.75,
                "value": 5,
            },
        },
        number={"font": {"color": color, "size": 28},
                "suffix": " : 1"},
    ))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ────────────────────────────────────────────
# トレンドサイクル可視化
# ────────────────────────────────────────────

def create_trend_cycle_chart(current_phase: str) -> go.Figure:
    """
    トレンドサイクル（上昇→横ばい→下落→横ばい）を
    環状に可視化する
    """
    phases = ["上昇", "横ばい\n(転換準備)", "下落", "横ばい\n(底固め)"]
    colors_phase = ["#00CC66", "#FFB800", "#FF3333", "#FFB800"]

    # 現在位置にあわせてハイライト
    highlight = []
    for ph in phases:
        plain = ph.replace("\n(転換準備)", "").replace("\n(底固め)", "")
        if plain == current_phase or (current_phase == "横ばい" and "横ばい" in ph):
            highlight.append(0.9)
        else:
            highlight.append(0.4)

    # 円グラフで環状表示
    fig = go.Figure(go.Pie(
        labels=phases,
        values=[1, 1, 1, 1],
        marker=dict(
            colors=[
                f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},{a})"
                for c, a in zip(colors_phase, highlight)
            ],
            line=dict(color="white", width=2)
        ),
        textinfo="label",
        textfont=dict(size=12, color="white"),
        hole=0.6,
        direction="clockwise",
        rotation=90,
        hoverinfo="label",
    ))

    # 中央に現在位置
    emoji_map = {"上昇": "🟢", "横ばい": "🟡", "下落": "🔴"}
    center_text = f"{emoji_map.get(current_phase, '⚪')}<br><b>{current_phase}</b>"
    fig.add_annotation(
        text=center_text,
        x=0.5, y=0.5,
        font=dict(size=16, color="white"),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(text="トレンドサイクル", font=dict(size=14, color="white"), x=0.5),
        paper_bgcolor="#0E1117",
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ────────────────────────────────────────────
# 移動平均線状態チャート
# ────────────────────────────────────────────

def create_ma_status_chart(df: pd.DataFrame) -> go.Figure:
    """MAの向きと並び状態をビジュアル表示"""
    if len(df) < 5:
        return go.Figure()

    last = df.iloc[-1]
    lookback = min(60, len(df))
    recent = df.iloc[-lookback:]

    # 各MAの値（NaNを除く）
    mas = {
        "5MA": (df["MA5"].dropna().iloc[-1] if not df["MA5"].isna().all() else None, COLORS["ma5"]),
        "20MA": (df["MA20"].dropna().iloc[-1] if not df["MA20"].isna().all() else None, COLORS["ma20"]),
        "60MA": (df["MA60"].dropna().iloc[-1] if not df["MA60"].isna().all() else None, COLORS["ma60"]),
        "400MA": (df["MA400"].dropna().iloc[-1] if not df["MA400"].isna().all() else None, COLORS["ma400"]),
    }

    fig = go.Figure()

    # 最新60本の終値
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["Close"],
        name="終値",
        line=dict(color="white", width=1),
        mode="lines"
    ))

    for name, (val, color) in mas.items():
        col = name.replace("MA", "MA")
        ma_col = "MA" + name.replace("MA", "").replace("MA", "")
        # keyからDataFrame列名に変換
        col_map = {"5MA": "MA5", "20MA": "MA20", "60MA": "MA60", "400MA": "MA400"}
        df_col = col_map.get(name)
        if df_col and df_col in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent.index, y=recent[df_col],
                name=name,
                line=dict(color=color, width=1.5),
                mode="lines"
            ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        height=200,
        margin=dict(l=5, r=50, t=5, b=5),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], side="right", tickformat=".2f"),
    )
    return fig


# ────────────────────────────────────────────
# ボリンジャーバンドの追加
# ────────────────────────────────────────────

def _add_bollinger_bands(fig: go.Figure, indicators: dict, df: pd.DataFrame) -> None:
    """ボリンジャーバンドをメインチャートに追加"""
    bb_upper = indicators.get("bb_upper")
    bb_mid = indicators.get("bb_mid")
    bb_lower = indicators.get("bb_lower")
    if bb_upper is None:
        return

    fig.add_trace(go.Scatter(
        x=df.index, y=bb_upper,
        name="BB上限(2σ)",
        line=dict(color=COLORS_BB["bb_upper"], width=1, dash="dot"),
        mode="lines",
        hovertemplate="BB上限: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=bb_lower,
        name="BB下限(2σ)",
        line=dict(color=COLORS_BB["bb_lower"], width=1, dash="dot"),
        fill="tonexty",
        fillcolor=COLORS_BB["bb_fill"],
        mode="lines",
        hovertemplate="BB下限: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=bb_mid,
        name="BB中心(20MA)",
        line=dict(color=COLORS_BB["bb_mid"], width=1),
        mode="lines",
        hovertemplate="BB中心: %{y:.2f}<extra></extra>",
    ), row=1, col=1)


# ────────────────────────────────────────────
# 一目均衡表の追加
# ────────────────────────────────────────────

def _add_ichimoku(fig: go.Figure, indicators: dict, df: pd.DataFrame) -> None:
    """一目均衡表をメインチャートに追加"""
    ichi_lines = indicators.get("ichi_lines", {})
    if not ichi_lines:
        return

    tenkan = ichi_lines.get("tenkan")
    kijun = ichi_lines.get("kijun")
    senkou_a = ichi_lines.get("senkou_a")
    senkou_b = ichi_lines.get("senkou_b")

    if tenkan is not None:
        fig.add_trace(go.Scatter(
            x=df.index, y=tenkan,
            name="転換線(9)",
            line=dict(color="#FF6B6B", width=1),
            mode="lines",
            hovertemplate="転換線: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    if kijun is not None:
        fig.add_trace(go.Scatter(
            x=df.index, y=kijun,
            name="基準線(26)",
            line=dict(color="#4ECDC4", width=1.5),
            mode="lines",
            hovertemplate="基準線: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    if senkou_a is not None and senkou_b is not None:
        fig.add_trace(go.Scatter(
            x=df.index, y=senkou_a,
            name="先行スパンA",
            line=dict(color="rgba(0,200,100,0.5)", width=1),
            mode="lines",
            hovertemplate="先行A: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=senkou_b,
            name="先行スパンB",
            line=dict(color="rgba(255,80,80,0.5)", width=1),
            fill="tonexty",
            fillcolor=COLORS_BB["ichimoku_up"],  # 簡略化：常に同色
            mode="lines",
            hovertemplate="先行B: %{y:.2f}<extra></extra>",
        ), row=1, col=1)


# ────────────────────────────────────────────
# RSI チャート
# ────────────────────────────────────────────

def create_rsi_chart(indicators: dict, df: pd.DataFrame) -> go.Figure:
    """RSI (14期間) インタラクティブチャート"""
    rsi_series = indicators.get("rsi_series")
    rsi_r = indicators.get("rsi")

    fig = go.Figure()
    if rsi_series is None or len(rsi_series) < 5:
        return fig

    # RSI ライン
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi_series,
        name="RSI(14)",
        line=dict(color="#9B59B6", width=2),
        mode="lines",
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ))

    # 過買い・過売りゾーン
    fig.add_hrect(y0=70, y1=100,
                  fillcolor="rgba(255,50,50,0.1)", line_width=0)
    fig.add_hrect(y0=0, y1=30,
                  fillcolor="rgba(0,200,100,0.1)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,80,80,0.6)", line_width=1,
                  annotation_text="買われすぎ(70)", annotation_font=dict(size=9))
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,100,0.6)", line_width=1,
                  annotation_text="売られすぎ(30)", annotation_position="right",
                  annotation_font=dict(size=9))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(150,150,150,0.4)", line_width=1)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        height=180,
        margin=dict(l=5, r=60, t=5, b=5),
        xaxis=dict(gridcolor=COLORS["grid"], showticklabels=False),
        yaxis=dict(gridcolor=COLORS["grid"], range=[0, 100],
                   side="right", tickvals=[30, 50, 70]),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────
# MACD チャート
# ────────────────────────────────────────────

def create_macd_chart(indicators: dict, df: pd.DataFrame) -> go.Figure:
    """MACDチャート（ライン + シグナル + ヒストグラム）"""
    macd_line = indicators.get("macd_line")
    signal_line = indicators.get("macd_signal")
    histogram = indicators.get("macd_hist")

    fig = go.Figure()
    if macd_line is None:
        return fig

    # ヒストグラム
    hist_colors = [
        "#00CC66" if float(v) >= 0 else "#FF4444"
        for v in histogram.fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=histogram,
        name="ヒストグラム",
        marker_color=hist_colors,
        opacity=0.7,
        hovertemplate="ヒスト: %{y:.3f}<extra></extra>",
    ))

    # MACDライン
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_line,
        name="MACD",
        line=dict(color="#00BFFF", width=1.5),
        mode="lines",
        hovertemplate="MACD: %{y:.3f}<extra></extra>",
    ))

    # シグナルライン
    fig.add_trace(go.Scatter(
        x=df.index, y=signal_line,
        name="Signal",
        line=dict(color="#FF6B6B", width=1.5),
        mode="lines",
        hovertemplate="Signal: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="rgba(150,150,150,0.4)", line_width=1)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        height=160,
        margin=dict(l=5, r=60, t=5, b=5),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], side="right", tickformat=".3f"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=9)),
    )
    return fig


# ────────────────────────────────────────────
# OBV チャート
# ────────────────────────────────────────────

def create_obv_chart(indicators: dict, df: pd.DataFrame) -> go.Figure:
    """OBV（累積出来高）チャート"""
    obv_series = indicators.get("obv_series")
    vol_r = indicators.get("volume")

    fig = go.Figure()
    if obv_series is None:
        return fig

    trend_color = "#00CC66" if (vol_r and vol_r.obv_trend == "上昇") else "#FF4444"
    fill_color = "rgba(0,204,102,0.13)" if (vol_r and vol_r.obv_trend == "上昇") else "rgba(255,68,68,0.13)"

    fig.add_trace(go.Scatter(
        x=df.index, y=obv_series,
        name="OBV",
        line=dict(color=trend_color, width=1.5),
        fill="tozeroy",
        fillcolor=fill_color,
        mode="lines",
        hovertemplate="OBV: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        height=140,
        margin=dict(l=5, r=60, t=5, b=5),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], side="right"),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────
# フィボナッチレベル表示（メインチャートに重ねる用）
# ────────────────────────────────────────────

def add_fibonacci_to_chart(fig: go.Figure, fib_r, current_price: float) -> None:
    """フィボナッチレトレースメントラインをチャートに追加"""
    if not fib_r or not fib_r.levels:
        return

    colors_fib = {
        0.236: "#FFE066",
        0.382: "#FFA500",
        0.500: "#FF6B6B",
        0.618: "#FF3399",  # 最重要（黄金比）
        0.786: "#CC00FF",
    }
    labels_fib = {
        0.236: "Fib 23.6%",
        0.382: "Fib 38.2%",
        0.500: "Fib 50.0%",
        0.618: "Fib 61.8% ★",
        0.786: "Fib 78.6%",
    }

    for ratio, price in fib_r.levels.items():
        color = colors_fib.get(ratio, "#AAAAAA")
        label = labels_fib.get(ratio, f"Fib {ratio:.1%}")
        dist = (price - current_price) / current_price * 100
        width = 2 if ratio == 0.618 else 1
        fig.add_hline(
            y=price,
            line_dash="dash",
            line_color=color,
            line_width=width,
            annotation_text=f"{label} {price:.2f} ({dist:+.1f}%)",
            annotation_position="right",
            annotation_font=dict(color=color, size=9),
            row=1, col=1
        )


# ────────────────────────────────────────────
# 複合スコアゲージ
# ────────────────────────────────────────────

def create_composite_gauge(composite) -> go.Figure:
    """複合スコアのゲージ表示（-10〜+10）"""
    score = composite.total_score
    color = composite.color

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "総合シグナルスコア", "font": {"size": 13, "color": "white"}},
        gauge={
            "axis": {"range": [-10, 10], "tickcolor": "white",
                     "tickfont": {"color": "white"},
                     "tickvals": [-10, -5, 0, 5, 10],
                     "ticktext": ["強売", "弱売", "中立", "弱買", "強買"]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [-10, -5], "color": "rgba(255,50,50,0.3)"},
                {"range": [-5, -2], "color": "rgba(255,100,100,0.2)"},
                {"range": [-2, 2],  "color": "rgba(150,150,150,0.1)"},
                {"range": [2, 5],   "color": "rgba(0,200,100,0.2)"},
                {"range": [5, 10],  "color": "rgba(0,200,100,0.3)"},
            ],
            "threshold": {
                "line": {"color": "#FFD700", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
            "shape": "angular",
        },
        number={"font": {"color": color, "size": 26},
                "suffix": f"  {composite.emoji}"},
    ))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        height=200,
        margin=dict(l=15, r=15, t=40, b=10),
    )
    return fig


# ────────────────────────────────────────────
# ATRベースの価格レンジ予測チャート
# ────────────────────────────────────────────

def create_atr_range_chart(atr_r, current_price: float, is_jp: bool = False) -> go.Figure:
    """ATRに基づく短期的な価格変動レンジを可視化"""
    if not atr_r or atr_r.atr_14 == 0:
        return go.Figure()

    currency = "¥" if is_jp else "$"
    atr = atr_r.atr_14
    levels = {
        "現在値": current_price,
        "+1 ATR": current_price + atr,
        "+2 ATR（損切り想定）": current_price + atr * 2,
        "+3 ATR": current_price + atr * 3,
        "+6 ATR（1:3目標）": current_price + atr * 6,
        "-1 ATR": current_price - atr,
        "-2 ATR（損切り基準）": current_price - atr * 2,
    }

    labels = list(levels.keys())
    prices = list(levels.values())
    colors_atr = [
        "#FFD700" if "現在値" in l
        else "#00CC66" if "+" in l
        else "#FF4444"
        for l in labels
    ]

    fig = go.Figure(go.Bar(
        x=[f"{currency}{p:,.1f}" for p in prices],
        y=labels,
        orientation="h",
        marker_color=colors_atr,
        text=[f"{currency}{p:,.2f}" for p in prices],
        textposition="outside",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        height=220,
        margin=dict(l=5, r=80, t=5, b=5),
        xaxis=dict(gridcolor="#1E2A35", showticklabels=False),
        yaxis=dict(gridcolor="#1E2A35"),
        showlegend=False,
    )
    return fig
