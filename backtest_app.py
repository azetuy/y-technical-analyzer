"""
バックテスト アプリ (backtest_app.py)

起動方法:
  cd tools/stock-analyzer
  streamlit run backtest_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from backtester import BacktestEngine, BacktestResult, run_batch_backtest, summarize_batch
from strategies import STRATEGIES, STRATEGY_NAMES, DEFAULT_US_TICKERS, DEFAULT_JP_TICKERS, get_strategy

# ────────────────────────────────────────────
# ページ設定
# ────────────────────────────────────────────
st.set_page_config(
    page_title="バックテスト｜Stock Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
    background: #1c2128; border-radius: 8px;
    padding: 12px 16px; margin: 4px 0;
  }
  .metric-good { color: #26a641; }
  .metric-bad  { color: #f85149; }
  .metric-neutral { color: #8b949e; }
  .strategy-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 12px;
    background: #1f6feb; color: white; margin-right: 4px;
  }
  .section-header {
    font-size: 18px; font-weight: 700;
    border-bottom: 1px solid #30363d; padding-bottom: 6px; margin: 16px 0 8px 0;
  }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# サイドバー設定
# ────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 バックテスト設定")

    # ── モード ──────────────────────────────
    mode = st.radio(
        "実行モード",
        ["単一銘柄テスト", "複数銘柄バッチ比較"],
        horizontal=True,
    )

    st.divider()

    # ── 銘柄入力 ────────────────────────────
    if mode == "単一銘柄テスト":
        ticker_input = st.text_input(
            "銘柄コード",
            value="NVDA",
            help="ティッカー(US: AAPL) または証券コード(JP: 7203)"
        )
    else:
        batch_preset = st.selectbox(
            "プリセット",
            ["US主要テック", "JP主要株", "カスタム"]
        )
        if batch_preset == "US主要テック":
            default_tickers = ", ".join(DEFAULT_US_TICKERS[:8])
        elif batch_preset == "JP主要株":
            default_tickers = ", ".join(DEFAULT_JP_TICKERS[:8])
        else:
            default_tickers = "AAPL, MSFT, NVDA, AMZN"

        ticker_input = st.text_area(
            "銘柄コード（カンマ区切り）",
            value=default_tickers,
            height=100,
        )

    # ── 戦略選択 ────────────────────────────
    st.divider()
    strategy_name = st.selectbox(
        "戦略",
        STRATEGY_NAMES,
        help="バックテストする戦略を選択"
    )
    cfg = get_strategy(strategy_name)
    with st.expander("戦略の詳細"):
        st.write(f"**{cfg.get('emoji', '')} {strategy_name}**")
        st.write(cfg.get("description", ""))
        st.caption(f"参考: {cfg.get('reference', '-')}")

    # ── パラメータ ───────────────────────────
    st.divider()
    period = st.select_slider(
        "バックテスト期間",
        options=["1y", "2y", "3y", "5y", "10y"],
        value="3y",
    )
    initial_capital = st.number_input(
        "初期資金",
        min_value=100_000,
        max_value=100_000_000,
        value=1_000_000,
        step=100_000,
        format="%d",
    )

    # ── 手数料・スリッページ ────────────────
    with st.expander("⚙️ 詳細設定"):
        commission = st.slider(
            "手数料（片道 %）", 0.0, 0.5, 0.1, 0.01, format="%.2f%%"
        ) / 100
        slippage = st.slider(
            "スリッページ（片道 %）", 0.0, 0.3, 0.05, 0.01, format="%.3f%%"
        ) / 100
        pos_size = st.slider(
            "ポジションサイズ（%）", 5, 50, int(cfg.get("position_size_pct", 10)), 1
        )
        max_hold = st.slider(
            "タイムストップ（営業日）", 5, 60, int(cfg.get("max_holding_days", 30)), 1
        )

    run_btn = st.button("▶ バックテスト実行", type="primary", use_container_width=True)


# ────────────────────────────────────────────
# ヘルパー: メトリクス表示
# ────────────────────────────────────────────
def _color(value: float, good_threshold: float = 0, invert: bool = False) -> str:
    """値に応じた色クラスを返す"""
    is_good = value > good_threshold if not invert else value < good_threshold
    return "metric-good" if is_good else "metric-bad"


def show_metrics(result: BacktestResult):
    """パフォーマンスメトリクスダッシュボードを表示"""
    r = result

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_color = "normal" if r.total_return_pct >= 0 else "inverse"
        st.metric("総リターン", f"{r.total_return_pct:+.1f}%",
                  f"年率 {r.annual_return_pct:+.1f}%", delta_color=delta_color)
    with col2:
        st.metric("勝率", f"{r.win_rate:.1f}%",
                  f"プロフィットファクター {r.profit_factor:.2f}")
    with col3:
        st.metric("最大ドローダウン", f"{r.max_drawdown_pct:.1f}%",
                  f"シャープ比 {r.sharpe_ratio:.2f}")
    with col4:
        st.metric("総トレード数", f"{r.total_trades}件",
                  f"平均保有 {r.avg_holding_days:.1f}日")

    # 詳細メトリクス
    with st.expander("詳細メトリクス"):
        cols = st.columns(3)
        metrics = [
            ("年率リターン",   f"{r.annual_return_pct:+.2f}%"),
            ("ソルティノ比",   f"{r.sortino_ratio:.2f}"),
            ("平均 R マルチプル", f"{r.avg_r_multiple:.2f}"),
            ("平均勝ちトレード",  f"{r.avg_win_pct:+.2f}%"),
            ("平均負けトレード",  f"{r.avg_loss_pct:+.2f}%"),
            ("最終資金",       f"¥{r.final_capital:,.0f}"),
        ]
        for idx, (label, val) in enumerate(metrics):
            cols[idx % 3].metric(label, val)


# ────────────────────────────────────────────
# ヘルパー: エクイティカーブ
# ────────────────────────────────────────────
def plot_equity_curve(result: BacktestResult, log_scale: bool = False) -> go.Figure:
    eq = result.equity_curve
    pct = (eq / result.initial_capital - 1) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=["エクイティカーブ", "ドローダウン (%)"],
        vertical_spacing=0.05,
    )

    # エクイティカーブ
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name="資産残高", line=dict(color="#26a641", width=2),
        fill="tozeroy", fillcolor="rgba(38, 166, 65, 0.08)",
    ), row=1, col=1)
    fig.add_hline(y=result.initial_capital, line_dash="dot",
                  line_color="#8b949e", row=1, col=1)

    # エントリー・エグジットマーカー
    closed = [t for t in result.trades if t.is_closed]
    if closed:
        entry_dates = [t.entry_date for t in closed]
        entry_equity = [float(eq.asof(d)) if d in eq.index or d >= eq.index[0] else None
                        for d in entry_dates]
        exit_dates  = [t.exit_date for t in closed]
        exit_equity  = [float(eq.asof(d)) if d in eq.index or d >= eq.index[0] else None
                        for d in exit_dates]

        fig.add_trace(go.Scatter(
            x=entry_dates, y=entry_equity,
            mode="markers", name="エントリー",
            marker=dict(symbol="triangle-up", size=8, color="#1f6feb"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=exit_dates, y=exit_equity,
            mode="markers", name="エグジット",
            marker=dict(symbol="triangle-down", size=8, color="#f85149"),
        ), row=1, col=1)

    # ドローダウン
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max * 100
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name="ドローダウン", line=dict(color="#f85149", width=1),
        fill="tozeroy", fillcolor="rgba(248, 81, 73, 0.15)",
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        showlegend=True,
    )
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)

    return fig


# ────────────────────────────────────────────
# ヘルパー: 月次リターンヒートマップ
# ────────────────────────────────────────────
def plot_monthly_returns(result: BacktestResult) -> go.Figure:
    pivot = result.monthly_returns()
    if pivot.empty:
        return go.Figure()

    months_jp = ["1月", "2月", "3月", "4月", "5月", "6月",
                 "7月", "8月", "9月", "10月", "11月", "12月"]
    cols_present = [m for m in months_jp if m in pivot.columns]
    data = pivot[cols_present].values
    years = pivot.index.tolist()

    max_abs = max(abs(data[~np.isnan(data)].max()), abs(data[~np.isnan(data)].min()), 0.01)

    text_arr = []
    for row in data:
        text_arr.append([f"{v:.1f}%" if not np.isnan(v) else "" for v in row])

    fig = go.Figure(go.Heatmap(
        z=data,
        x=cols_present,
        y=[str(y) for y in years],
        text=text_arr,
        texttemplate="%{text}",
        colorscale=[[0, "#f85149"], [0.5, "#161b22"], [1, "#26a641"]],
        zmin=-max_abs,
        zmax=max_abs,
        showscale=True,
        colorbar=dict(title="リターン(%)"),
    ))
    fig.update_layout(
        title="月次リターン ヒートマップ (%)",
        template="plotly_dark",
        height=max(200, len(years) * 35 + 100),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ────────────────────────────────────────────
# ヘルパー: 損益分布ヒストグラム
# ────────────────────────────────────────────
def plot_pnl_distribution(result: BacktestResult) -> go.Figure:
    pnls = [t.pnl_pct for t in result.closed_trades]
    if not pnls:
        return go.Figure()

    colors = ["#26a641" if p > 0 else "#f85149" for p in pnls]

    fig = go.Figure(go.Histogram(
        x=pnls,
        nbinsx=30,
        name="損益分布",
        marker=dict(
            color=["#26a641" if p > 0 else "#f85149" for p in pnls],
            line=dict(width=0),
        ),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#8b949e")
    fig.add_vline(x=float(np.mean(pnls)), line_dash="dot",
                  line_color="#1f6feb", annotation_text=f"平均 {np.mean(pnls):.1f}%")

    fig.update_layout(
        title="損益分布 (%)",
        xaxis_title="損益 (%)",
        yaxis_title="トレード数",
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────
# ヘルパー: トレードリスト
# ────────────────────────────────────────────
def show_trade_list(result: BacktestResult):
    trades = result.closed_trades
    if not trades:
        st.info("クローズドトレードがありません。")
        return

    rows = []
    for t in trades:
        rows.append({
            "ID": t.trade_id,
            "エントリー日": t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "-",
            "エグジット日": t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "-",
            "エントリー価格": f"{t.entry_price:.2f}",
            "エグジット価格": f"{t.exit_price:.2f}" if t.exit_price else "-",
            "損益(%)": f"{t.pnl_pct:+.2f}%",
            "Rマルチプル": f"{t.r_multiple:.2f}R",
            "保有日数": t.holding_days,
            "出口理由": t.exit_reason,
            "シグナル": t.entry_signal_detail,
        })

    df = pd.DataFrame(rows)

    def color_pnl(val):
        if "+" in str(val):
            return "color: #26a641"
        elif "-" in str(val):
            return "color: #f85149"
        return ""

    styled = df.style.map(color_pnl, subset=["損益(%)"])
    st.dataframe(styled, use_container_width=True, height=400)


# ────────────────────────────────────────────
# ヘルパー: 出口理由の円グラフ
# ────────────────────────────────────────────
def plot_exit_reasons(result: BacktestResult) -> go.Figure:
    counts = result.exit_reason_counts
    if not counts:
        return go.Figure()

    colors_map = {
        "損切り": "#f85149",
        "利確": "#26a641",
        "タイムストップ": "#e3b341",
        "シグナル反転（5MA割れ）": "#58a6ff",
        "MACDデッドクロス": "#bc8cff",
        "スコア悪化": "#ffa657",
        "RSI買われすぎ": "#3fb950",
        "期間終了": "#8b949e",
    }

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [colors_map.get(l, "#8b949e") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hole=0.4,
    ))
    fig.update_layout(
        title="出口理由の内訳",
        template="plotly_dark",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


# ────────────────────────────────────────────
# ヘルパー: バッチ比較グラフ
# ────────────────────────────────────────────
def plot_batch_comparison(summary_df: pd.DataFrame) -> go.Figure:
    """複数銘柄のリターン・勝率バブルチャート"""
    df = summary_df.copy()
    df = df[df["プロフィットファクター"] != "∞"].copy()
    df["プロフィットファクター"] = pd.to_numeric(df["プロフィットファクター"], errors="coerce")

    fig = go.Figure(go.Scatter(
        x=df["勝率(%)"],
        y=df["総リターン(%)"],
        mode="markers+text",
        text=df["銘柄"],
        textposition="top center",
        marker=dict(
            size=df["総トレード数"].clip(lower=1) * 3,
            color=df["シャープ比"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="シャープ比"),
            opacity=0.8,
        ),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e")
    fig.add_vline(x=50, line_dash="dash", line_color="#8b949e")

    fig.update_layout(
        title="銘柄別 勝率 vs 総リターン（バブルサイズ = トレード数、色 = シャープ比）",
        xaxis_title="勝率 (%)",
        yaxis_title="総リターン (%)",
        template="plotly_dark",
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_batch_equity_overlay(all_results: dict[str, BacktestResult]) -> go.Figure:
    """複数銘柄のエクイティカーブを重ね描き"""
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for idx, (ticker, result) in enumerate(all_results.items()):
        eq = result.equity_curve
        pct = (eq / result.initial_capital - 1) * 100
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.values,
            name=ticker.split("（")[0],
            line=dict(color=color, width=1.5),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#8b949e")
    fig.update_layout(
        title="エクイティカーブ比較（リターン %）",
        yaxis_title="リターン (%)",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ────────────────────────────────────────────
# メインコンテンツ
# ────────────────────────────────────────────
st.title("🧪 バックテスト システム")
st.caption(f"戦略: **{strategy_name}** ｜ 期間: **{period}** ｜ 初期資金: **¥{initial_capital:,}**")

# ─────────────────
# 実行ロジック
# ─────────────────
if run_btn:
    engine_cfg = get_strategy(strategy_name)

    # エンジン設定（サイドバーの詳細設定で上書き）
    engine_kwargs = dict(
        commission_rate=commission,
        slippage_rate=slippage,
        position_size_pct=pos_size,
        max_holding_days=max_hold,
    )

    # ── 単一銘柄モード ─────────────────────────
    if mode == "単一銘柄テスト":
        with st.spinner(f"バックテスト実行中: {ticker_input}..."):
            engine = BacktestEngine(**engine_kwargs)
            try:
                result = engine.run(
                    ticker_input, strategy_name, engine_cfg,
                    period=period, initial_capital=initial_capital,
                )
                st.session_state["bt_result"] = result
                st.session_state["bt_mode"] = "single"
                st.success(f"✅ 完了: {result.ticker}")
            except Exception as e:
                st.error(f"エラー: {e}")
                st.stop()

    # ── バッチモード ───────────────────────────
    else:
        raw_tickers = [t.strip() for t in ticker_input.replace("、", ",").split(",") if t.strip()]
        if not raw_tickers:
            st.error("銘柄コードを入力してください。")
            st.stop()

        progress = st.progress(0, text="バッチバックテスト実行中...")
        all_results = {}
        for idx, t in enumerate(raw_tickers):
            progress.progress((idx + 1) / len(raw_tickers), text=f"処理中: {t}")
            try:
                engine = BacktestEngine(**engine_kwargs)
                r = engine.run(t, strategy_name, engine_cfg,
                               period=period, initial_capital=initial_capital)
                all_results[t] = r
            except Exception as e:
                st.warning(f"スキップ [{t}]: {e}")

        progress.empty()
        st.session_state["bt_result"] = all_results
        st.session_state["bt_mode"] = "batch"
        st.success(f"✅ 完了: {len(all_results)}/{len(raw_tickers)} 銘柄")


# ─────────────────
# 結果表示
# ─────────────────
if "bt_result" not in st.session_state:
    # 初期画面
    st.markdown("""
    ---
    ### 使い方

    1. サイドバーで **銘柄コード・戦略・期間** を設定
    2. **▶ バックテスト実行** ボタンをクリック
    3. 結果（エクイティカーブ、月次リターン、トレードリスト等）を確認

    ---
    ### 実装済み戦略一覧
    """)
    for name, cfg in STRATEGIES.items():
        st.markdown(
            f"**{cfg.get('emoji','')} {name}**  \n"
            f"{cfg.get('description','')}"
        )
        st.caption(f"参考: {cfg.get('reference', '-')}")
        st.divider()
    st.stop()


bt_mode = st.session_state.get("bt_mode", "single")

# ═══════════════════════════════════════
# 単一銘柄結果
# ═══════════════════════════════════════
if bt_mode == "single":
    result: BacktestResult = st.session_state["bt_result"]

    # ── ヘッダー ─────────────────────────
    st.markdown(f"## 📊 {result.ticker}")
    st.caption(f"戦略: {result.strategy_name} ｜ 期間: {result.period}")

    # ── パフォーマンスメトリクス ──────────
    st.markdown('<div class="section-header">パフォーマンスサマリー</div>', unsafe_allow_html=True)
    show_metrics(result)

    # ── エクイティカーブ ──────────────────
    st.markdown('<div class="section-header">エクイティカーブ</div>', unsafe_allow_html=True)
    log_scale = st.checkbox("対数スケール", value=False, key="log_scale_single")
    st.plotly_chart(plot_equity_curve(result, log_scale), use_container_width=True)

    # ── 月次リターン ──────────────────────
    st.markdown('<div class="section-header">月次リターン</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_monthly_returns(result), use_container_width=True)

    # ── 損益分布 & 出口理由 ───────────────
    col_hist, col_pie = st.columns(2)
    with col_hist:
        st.markdown('<div class="section-header">損益分布</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_pnl_distribution(result), use_container_width=True)
    with col_pie:
        st.markdown('<div class="section-header">出口理由</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_exit_reasons(result), use_container_width=True)

    # ── トレードリスト ────────────────────
    st.markdown('<div class="section-header">トレード履歴</div>', unsafe_allow_html=True)
    show_trade_list(result)

    # ── CSVエクスポート ───────────────────
    if result.closed_trades:
        rows = []
        for t in result.closed_trades:
            rows.append({
                "ticker": result.ticker,
                "strategy": result.strategy_name,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl_pct": round(t.pnl_pct, 4),
                "r_multiple": round(t.r_multiple, 4),
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
                "signal": t.entry_signal_detail,
            })
        csv = pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 トレードデータをCSVでダウンロード",
            data=csv,
            file_name=f"backtest_{result.ticker.split('（')[0]}_{strategy_name}.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════
# バッチ結果
# ═══════════════════════════════════════
elif bt_mode == "batch":
    all_results: dict[str, BacktestResult] = st.session_state["bt_result"]

    if not all_results:
        st.error("結果がありません。")
        st.stop()

    summary = summarize_batch(all_results)

    st.markdown("## 📊 バッチバックテスト結果")
    st.caption(f"戦略: {strategy_name} ｜ 期間: {period} ｜ {len(all_results)} 銘柄")

    # ── サマリーテーブル ──────────────────
    st.markdown('<div class="section-header">銘柄別パフォーマンス比較</div>', unsafe_allow_html=True)

    def color_returns(val):
        try:
            v = float(val)
            return "color: #26a641" if v > 0 else "color: #f85149"
        except:
            return ""

    styled_summary = summary.style\
        .map(color_returns, subset=["総リターン(%)", "年率リターン(%)"])\
        .format(precision=2)
    st.dataframe(styled_summary, use_container_width=True)

    # ── バブルチャート ────────────────────
    st.markdown('<div class="section-header">勝率 vs 総リターン（バブルチャート）</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_batch_comparison(summary), use_container_width=True)

    # ── エクイティカーブ重ね描き ──────────
    st.markdown('<div class="section-header">エクイティカーブ比較</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_batch_equity_overlay(all_results), use_container_width=True)

    # ── 個別銘柄詳細 ──────────────────────
    st.markdown('<div class="section-header">個別銘柄の詳細</div>', unsafe_allow_html=True)
    ticker_select = st.selectbox(
        "銘柄を選択",
        list(all_results.keys()),
    )
    if ticker_select and ticker_select in all_results:
        sel_result = all_results[ticker_select]
        show_metrics(sel_result)

        tab1, tab2, tab3 = st.tabs(["エクイティカーブ", "月次リターン", "トレードリスト"])
        with tab1:
            log_scale_b = st.checkbox("対数スケール", value=False, key="log_scale_batch")
            st.plotly_chart(plot_equity_curve(sel_result, log_scale_b), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_monthly_returns(sel_result), use_container_width=True)
        with tab3:
            show_trade_list(sel_result)

    # ── CSVエクスポート ───────────────────
    csv = summary.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 サマリーをCSVでダウンロード",
        data=csv,
        file_name=f"batch_backtest_{strategy_name}.csv",
        mime="text/csv",
    )
