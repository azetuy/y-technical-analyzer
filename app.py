"""
Y式 株価パターン分析ツール
Streamlit メインアプリ

起動方法:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from analyzer import (
    run_full_analysis, get_market_overview, calc_risk_reward,
    detect_range_zones, AnalysisResult
)
from chart import (
    create_main_chart, create_rr_gauge,
    create_trend_cycle_chart, create_ma_status_chart,
    create_rsi_chart, create_macd_chart, create_obv_chart,
    add_fibonacci_to_chart, create_composite_gauge, create_atr_range_chart,
)

# ────────────────────────────────────────────
# ページ設定
# ────────────────────────────────────────────

st.set_page_config(
    page_title="Y式 株価パターン分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（ダークテーマ最適化）
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; }

    /* カード風コンポーネント */
    .metric-card {
        background: #1E2A35;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2D3F4F;
        margin-bottom: 8px;
    }
    .metric-card h4 { color: #8899AA; font-size: 12px; margin: 0 0 4px; }
    .metric-card .value { font-size: 22px; font-weight: bold; margin: 0; }

    /* 買鉄条件チェックリスト */
    .kaitetsu-condition {
        display: flex;
        align-items: center;
        padding: 6px 10px;
        border-radius: 6px;
        margin: 3px 0;
        font-size: 13px;
    }
    .kaitetsu-ok { background: rgba(0,200,100,0.15); border-left: 3px solid #00CC66; }
    .kaitetsu-ng { background: rgba(255,50,50,0.15); border-left: 3px solid #FF3333; }

    /* 支持線・抵抗線テーブル */
    .sr-resistance { color: #FF6666; }
    .sr-support { color: #6699FF; }

    /* ヘッダー */
    .section-header {
        color: #8899AA;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 1px solid #2D3F4F;
        padding-bottom: 4px;
        margin-bottom: 10px;
    }

    /* 警告バナー */
    .warning-box {
        background: rgba(255,165,0,0.15);
        border: 1px solid #FFB800;
        border-radius: 8px;
        padding: 10px 14px;
        color: #FFB800;
        font-size: 13px;
    }

    /* パターン検出カード */
    .pattern-detected {
        background: rgba(0,200,100,0.1);
        border: 1px solid #00CC66;
        border-radius: 8px;
        padding: 10px;
        margin: 4px 0;
    }
    .pattern-none {
        background: rgba(100,100,100,0.1);
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
        margin: 4px 0;
        color: #666;
    }

    /* Streamlitデフォルト要素 */
    .stMetric label { color: #8899AA !important; font-size: 12px !important; }
    .stMetric .metric-value { font-size: 20px !important; }
    div[data-testid="metric-container"] {
        background: #1E2A35;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #2D3F4F;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# サイドバー
# ────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Y式 分析ツール")
    st.markdown("---")

    # 銘柄入力
    st.markdown("### 銘柄入力")
    ticker_input = st.text_input(
        "ティッカー / 証券コード",
        value="AAPL",
        placeholder="例: AAPL, 7203, NVDA",
        help="米国株はティッカー（AAPL等）、日本株は証券コード（7203等）を入力"
    )

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox(
            "期間",
            ["3mo", "6mo", "1y", "2y", "5y"],
            index=2,
            format_func=lambda x: {
                "3mo": "3ヶ月", "6mo": "6ヶ月", "1y": "1年",
                "2y": "2年", "5y": "5年"
            }[x]
        )
    with col2:
        interval = st.selectbox(
            "時間軸",
            ["1d", "1wk", "1mo"],
            index=0,
            format_func=lambda x: {"1d": "日足", "1wk": "週足", "1mo": "月足"}[x]
        )

    analyze_btn = st.button("🔍 分析する", type="primary", use_container_width=True)

    st.markdown("---")

    # 表示設定
    st.markdown("### 表示設定")
    show_sr = st.checkbox("支持線・抵抗線", value=True)
    show_gaps = st.checkbox("ギャップ（窓）", value=True)
    show_highs_lows = st.checkbox("Y式高値安値", value=True)
    show_kaitetsu = st.checkbox("買鉄シグナル", value=True)
    show_bb = st.checkbox("ボリンジャーバンド", value=True)
    show_ichimoku = st.checkbox("一目均衡表", value=False)
    show_fib = st.checkbox("フィボナッチ", value=False)

    st.markdown("---")

    # 市場地合い
    st.markdown("### 市場地合い")
    with st.spinner("取得中..."):
        market_data = get_market_overview()

    for name, data in market_data.items():
        direction = "▲" if data["up"] else "▼"
        color_style = "color: #00CC66;" if data["up"] else "color: #FF3333;"
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between;
            padding:4px 0; border-bottom:1px solid #1E2A35; font-size:12px;">
            <span style="color:#8899AA;">{name}</span>
            <span style="{color_style}">{direction} {data['price']:.2f} ({data['change_pct']:+.1f}%)</span>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Yさんの格言
    st.markdown("### Yさんの言葉")
    quotes = [
        "「早くお手上げした方が勝ち」",
        "「事実（ライン）に降参する勇気を持つ」",
        "「損切りは防衛、負けではない」",
        "「月足が全ての土台」",
        "「肉を切らせて骨を断つ」",
        "「見込み利益1:5以上でやる価値あり」",
    ]
    import random
    st.markdown(f"*{random.choice(quotes)}*")


# ────────────────────────────────────────────
# セッション状態
# ────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None

if analyze_btn and ticker_input:
    with st.spinner(f"「{ticker_input}」を分析中..."):
        result = run_full_analysis(ticker_input, period=period, interval=interval)
        if result is None:
            st.error(f"「{ticker_input}」のデータを取得できませんでした。ティッカーコードを確認してください。")
        else:
            st.session_state.result = result


# ────────────────────────────────────────────
# メインコンテンツ
# ────────────────────────────────────────────

result: AnalysisResult = st.session_state.result

if result is None:
    # ウェルカム画面
    st.markdown("## 📈 Y式 株価パターン分析ツール")
    st.markdown("""
    **Yさん「チャートの詠み方 極」に基づくテクニカル分析ツールです。**

    ---

    ### このツールでできること

    | 機能 | 説明 |
    |------|------|
    | 🟢 **トレンド判定** | 上昇/横ばい/下落の3フェーズを20ヶ月線で判定 |
    | 🎯 **買鉄チェック** | Y式5条件を自動チェック |
    | 📏 **支持線・抵抗線** | 高値安値ラインを自動抽出・反応回数で強度評価 |
    | 🕳️ **ギャップ検出** | 窓（ギャップ）の自動検出・表示 |
    | 📊 **チャートパターン** | カップ・ウィズ・ハンドル、三角収束等を検出 |
    | ⚖️ **R:Rシミュレーター** | リスクリワード比を計算（Y式推奨1:5以上） |

    ---

    ### 使い方
    1. 左サイドバーでティッカーコードまたは証券コードを入力
    2. 分析期間と時間軸を選択
    3. 「🔍 分析する」ボタンを押す

    **例:** AAPL（アップル）、NVDA（エヌビディア）、7203（トヨタ）、6758（ソニー）
    """)

    st.markdown("---")
    st.markdown("""
    ### Y式の核心

    > **「月足（20ヶ月線）が全ての土台。日足の判断は月足の支配下にある。」**

    - 相場は `上昇 → 横ばい → 下落 → 横ばい` の順に循環する
    - **買鉄**は5条件が揃ったときのみ機械的に入る
    - 1条件でも崩れたら**機械的に損切り**
    - 目標は**リスクリワード 1:5以上**の損小利大
    """)

else:
    # ── ヘッダー情報 ─────────────────────────────
    ticker_display = result.ticker.replace(".T", "（日本株）")

    st.markdown(f"## {result.company_name}")
    st.markdown(f"**{ticker_display}** | {datetime.now().strftime('%Y-%m-%d %H:%M')} 更新")

    # ── トップメトリクス行 ─────────────────────────
    col_price, col_trend, col_perf, col_kaitetsu, col_stoploss = st.columns(5)

    with col_price:
        price_str = f"¥{result.current_price:,.0f}" if ".T" in result.ticker else f"${result.current_price:.2f}"
        st.metric("現在値", price_str)

    with col_trend:
        st.metric(
            "トレンド",
            f"{result.trend.emoji} {result.trend.phase}",
            help="20ヶ月線（400日線）ベースで判定"
        )

    with col_perf:
        po = result.trend.perfect_order
        po_icon = "🌟" if "最強" in po else "👀" if "監視" in po else "⚡" if "エントリー" in po else "💀" if "最弱" in po else "➖"
        st.metric(
            "MAの状態",
            f"{po_icon} {po[:8] if po != 'なし' else 'なし'}",
            help="パーフェクトオーダー判定"
        )

    with col_kaitetsu:
        k = result.kaitetsu
        st.metric(
            "買鉄スコア",
            f"{k.score}/5",
            delta="シグナル！" if k.is_valid else None,
            delta_color="normal" if k.is_valid else "off"
        )

    with col_stoploss:
        if result.stop_loss_price > 0:
            sl_str = f"¥{result.stop_loss_price:,.0f}" if ".T" in result.ticker else f"${result.stop_loss_price:.2f}"
            sl_dist = (result.current_price - result.stop_loss_price) / result.current_price * 100
            st.metric(
                "損切り基準（5MA）",
                sl_str,
                delta=f"{sl_dist:.1f}%",
                delta_color="inverse",
                help="5MA終値割れで機械的に損切り（Y式）"
            )

    # ── 総合シグナルバナー ─────────────────────
    if result.indicators.get("composite"):
        comp = result.indicators["composite"]
        banner_color = comp.color
        verdict = comp.verdict
        score = comp.total_score
        st.markdown(
            f"""<div style="background:linear-gradient(90deg, {banner_color}22, transparent);
            border-left:4px solid {banner_color}; border-radius:8px;
            padding:10px 16px; margin:8px 0; display:flex;
            justify-content:space-between; align-items:center;">
            <div>
              <span style="color:{banner_color}; font-size:20px; font-weight:bold;">
                {comp.emoji} {verdict}
              </span>
              <span style="color:#8899AA; font-size:12px; margin-left:12px;">
                （スコア: {score:+.1f} / ±10）
              </span>
            </div>
            <div style="color:#8899AA; font-size:11px;">
              買いシグナル {len(comp.buy_signals)}個 / 売りシグナル {len(comp.sell_signals)}個
            </div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── メインチャート ─────────────────────────────
    with st.container():
        fig = create_main_chart(
            result,
            show_sr=show_sr,
            show_gaps=show_gaps,
            show_highs_lows=show_highs_lows,
            show_kaitetsu=show_kaitetsu,
            show_bb=show_bb,
            show_ichimoku=show_ichimoku,
        )
        # フィボナッチを追加（チェック時）
        if show_fib and result.indicators.get("fibonacci"):
            add_fibonacci_to_chart(fig, result.indicators["fibonacci"], result.current_price)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    st.markdown("---")

    # ── 下段3列レイアウト ──────────────────────────
    col_left, col_center, col_right = st.columns([1, 1, 1])

    # ────── 左列: Y式分析 ───────────────────
    with col_left:
        st.markdown('<div class="section-header">Y式 分析</div>', unsafe_allow_html=True)

        # トレンドサイクル
        cycle_fig = create_trend_cycle_chart(result.trend.phase)
        st.plotly_chart(cycle_fig, use_container_width=True)

        # トレンド詳細
        t = result.trend
        with st.expander("📊 トレンド詳細", expanded=True):
            st.markdown(f"""
            | 項目 | 状態 |
            |------|------|
            | 400日線の向き | {'↗ 上向き' if t.ma400_up else '↘ 下向き'} |
            | 株価 vs 400日線 | {'上' if t.price_above_ma400 else '下'} |
            | 5MA傾き | {'↗' if t.ma5_slope > 0 else '↘'} |
            | 20MA傾き | {'↗' if t.ma20_slope > 0 else '↘'} |
            | 60MA傾き | {'↗' if t.ma60_slope > 0 else '↘'} |
            | 20MA乗り継続 | {t.days_above_ma20}日 / 目安20日 |
            """)

            if t.days_above_ma20 >= 18:
                st.markdown('<div class="warning-box">⚠️ 20MA乗りが20日近い。賞味期限に注意。</div>',
                           unsafe_allow_html=True)

        # MTF（マルチタイムフレーム）ヒント
        with st.expander("🔭 MTF（時間軸変換）ヒント"):
            st.markdown("""
            | 日足MA | 週足相当 |
            |--------|---------|
            | **20MA** | 週足5MA相当 |
            | **100MA** | 週足20MA相当 |
            | **400MA** | 月足20MA相当 |

            *日足で全て見ることで、週足切り替え不要で先読み可能（Y式）*
            """)

    # ────── 中列: 買鉄チェック & パターン ──────────
    with col_center:
        st.markdown('<div class="section-header">買鉄チェックリスト</div>', unsafe_allow_html=True)

        k = result.kaitetsu

        # スコアバー
        score_color = "#00CC66" if k.score == 5 else "#FFB800" if k.score >= 3 else "#FF3333"
        st.markdown(
            f"""<div style="background:#1E2A35; border-radius:10px; padding:12px;
            border:2px solid {score_color}; margin-bottom:12px;">
            <div style="font-size:18px; font-weight:bold; color:{score_color};">{k.label}</div>
            <div style="background:#333; border-radius:4px; height:8px; margin-top:8px;">
                <div style="background:{score_color}; width:{k.score/5*100}%; height:8px; border-radius:4px;"></div>
            </div>
            </div>""",
            unsafe_allow_html=True
        )

        conditions = [
            (k.cond1_ma_order, "条件1: 5MA > 20MA（上昇の並び）"),
            (k.cond2_ma20_up, "条件2: 20MAが上向き（トレンド確認）"),
            (k.cond3_ma5_turn_up, "条件3: 5MAが下→上に転換（反転タイミング）"),
            (k.cond4_body_above_ma5, "条件4: 実体の50%以上が5MA上（反転確認）"),
            (k.cond5_near_ma20, "条件5: 20MAに接近/タッチ（調整の証明）"),
        ]

        for ok, label in conditions:
            icon = "✅" if ok else "❌"
            css_class = "kaitetsu-ok" if ok else "kaitetsu-ng"
            st.markdown(
                f'<div class="kaitetsu-condition {css_class}">{icon} {label}</div>',
                unsafe_allow_html=True
            )

        if k.is_valid:
            st.success("🚀 買鉄シグナル！5条件すべて達成。\n損切り：5MA終値割れで機械的に実行。")
        elif k.score >= 3:
            missing = sum(1 for ok, _ in conditions if not ok)
            st.warning(f"⚠️ 買鉄候補 — あと{missing}条件待ち。")

        st.markdown("---")

        # チャートパターン
        st.markdown('<div class="section-header">チャートパターン</div>', unsafe_allow_html=True)

        for pat in result.patterns:
            if pat.detected:
                st.markdown(
                    f'<div class="pattern-detected">🎯 <b>{pat.name}</b><br>'
                    f'<small style="color:#8899AA;">{pat.description}</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="pattern-none">— {pat.name}: 非検出</div>',
                    unsafe_allow_html=True
                )

        # 横ばい帯
        df = result.df
        range_zones = detect_range_zones(df)
        if range_zones:
            st.markdown(f"📦 **横ばい帯**: {len(range_zones)}箇所検出（直近20本）")

    # ────── 右列: 支持線・抵抗線 & R:R ────────────
    with col_right:
        st.markdown('<div class="section-header">支持線・抵抗線</div>', unsafe_allow_html=True)

        sr = result.support_resistance
        current_price = result.current_price

        # 抵抗線（現在値より上）
        resistances = [lvl for lvl in sr if lvl.price > current_price and lvl.level_type == "resistance"]
        resistances = sorted(resistances, key=lambda x: x.price)[:5]  # 近い5本

        # 支持線（現在値より下）
        supports = [lvl for lvl in sr if lvl.price < current_price and lvl.level_type == "support"]
        supports = sorted(supports, key=lambda x: x.price, reverse=True)[:5]  # 近い5本

        # 抵抗線テーブル
        st.markdown("**🔴 抵抗線（上値）**")
        if resistances:
            for lvl in reversed(resistances):
                dist = (lvl.price - current_price) / current_price * 100
                strength_icon = "🔴" if lvl.strength == "強" else "🟠" if lvl.strength == "中" else "🟡"
                ticker_suffix = ".T" in result.ticker
                price_str = f"¥{lvl.price:,.0f}" if ticker_suffix else f"${lvl.price:.2f}"
                st.markdown(
                    f'{strength_icon} **{price_str}** (+{dist:.1f}%) — {lvl.touch_count}回 {lvl.source}',
                )
        else:
            st.markdown("*抵抗線なし（最高値付近の可能性）*")

        # 現在値
        cp_str = f"¥{current_price:,.0f}" if ".T" in result.ticker else f"${current_price:.2f}"
        st.markdown(
            f'<div style="background:#2D3F4F; border-radius:6px; padding:6px 10px; '
            f'text-align:center; margin:8px 0; border:1px solid #FFD700; color:#FFD700; font-weight:bold;">'
            f'━━ 現在値 {cp_str} ━━</div>',
            unsafe_allow_html=True
        )

        # 支持線テーブル
        st.markdown("**🔵 支持線（下値）**")
        if supports:
            for lvl in supports:
                dist = (current_price - lvl.price) / current_price * 100
                strength_icon = "🔵" if lvl.strength == "強" else "🟦" if lvl.strength == "中" else "🔷"
                ticker_suffix = ".T" in result.ticker
                price_str = f"¥{lvl.price:,.0f}" if ticker_suffix else f"${lvl.price:.2f}"
                st.markdown(
                    f'{strength_icon} **{price_str}** (-{dist:.1f}%) — {lvl.touch_count}回 {lvl.source}',
                )
        else:
            st.markdown("*支持線なし（底値付近の可能性）*")

        st.markdown("---")

        # ── リスクリワードシミュレーター ────────────
        st.markdown('<div class="section-header">R:R シミュレーター</div>', unsafe_allow_html=True)
        st.markdown("*Y式推奨: リスクリワード 1:5以上*")

        is_jp = ".T" in result.ticker
        currency = "¥" if is_jp else "$"

        col_rr1, col_rr2, col_rr3 = st.columns(3)
        with col_rr1:
            entry = st.number_input(
                "エントリー価格",
                value=float(f"{current_price:.2f}"),
                min_value=0.0, step=0.1 if not is_jp else 1.0,
                format="%.2f" if not is_jp else "%.0f"
            )
        with col_rr2:
            stop_loss = st.number_input(
                "損切り価格（5MA）",
                value=float(f"{result.stop_loss_price:.2f}") if result.stop_loss_price > 0 else float(f"{current_price * 0.95:.2f}"),
                min_value=0.0, step=0.1 if not is_jp else 1.0,
                format="%.2f" if not is_jp else "%.0f"
            )
        with col_rr3:
            # 次の抵抗線を目標として提案
            default_target = resistances[0].price if resistances else current_price * 1.10
            target = st.number_input(
                "目標価格",
                value=float(f"{default_target:.2f}"),
                min_value=0.0, step=0.1 if not is_jp else 1.0,
                format="%.2f" if not is_jp else "%.0f"
            )

        if entry > 0 and stop_loss > 0 and target > 0:
            rr = calc_risk_reward(entry, stop_loss, target)
            if rr["ratio"] > 0:
                gauge_fig = create_rr_gauge(rr["ratio"])
                st.plotly_chart(gauge_fig, use_container_width=True)

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    risk_str = f"{currency}{rr['risk']:.2f}" if not is_jp else f"{currency}{rr['risk']:.0f}"
                    st.metric("リスク（損失）", risk_str)
                with col_r2:
                    reward_str = f"{currency}{rr['reward']:.2f}" if not is_jp else f"{currency}{rr['reward']:.0f}"
                    st.metric("リワード（利益）", reward_str)

                if rr["ok"]:
                    st.success(rr["comment"])
                else:
                    st.warning(rr["comment"])

                # ポジションサイズ
                with st.expander("💰 ポジションサイズ逆算"):
                    allowed_loss = st.number_input(
                        "許容損失額",
                        value=10000 if is_jp else 100.0,
                        step=1000 if is_jp else 10.0,
                        format="%.0f" if is_jp else "%.2f"
                    )
                    if rr["risk"] > 0:
                        shares = allowed_loss / rr["risk"]
                        total = shares * entry
                        st.markdown(f"""
                        - 推奨株数: **{shares:.1f}株**
                        - 投資総額: **{currency}{total:,.0f}**
                        - 最大損失: **{currency}{shares * rr['risk']:,.0f}**
                        - 期待利益: **{currency}{shares * rr['reward']:,.0f}**
                        """)

    # ════════════════════════════════════════════
    # テクニカル指標パネル（一般指標）
    # ════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 📊 テクニカル指標分析")

    if result.indicators:
        ind = result.indicators
        rsi_r = ind.get("rsi")
        macd_r = ind.get("macd")
        bb_r = ind.get("bb")
        ichi_r = ind.get("ichi")
        mom_r = ind.get("momentum")
        vol_r = ind.get("volume")
        atr_r = ind.get("atr")
        fib_r = ind.get("fibonacci")
        cross_r = ind.get("cross")
        comp = ind.get("composite")

        # ── 複合スコアと信号リスト ──────────────────
        col_gauge, col_signals = st.columns([1, 2])
        with col_gauge:
            if comp:
                gauge_fig = create_composite_gauge(comp)
                st.plotly_chart(gauge_fig, use_container_width=True)

        with col_signals:
            if comp:
                tab_buy, tab_sell, tab_neutral = st.tabs(
                    [f"🟢 買いシグナル ({len(comp.buy_signals)})",
                     f"🔴 売りシグナル ({len(comp.sell_signals)})",
                     f"⚪ 中立 ({len(comp.neutral_signals)})"]
                )
                with tab_buy:
                    if comp.buy_signals:
                        for s in comp.buy_signals:
                            st.markdown(f"- ✅ {s}")
                    else:
                        st.markdown("*買いシグナルなし*")
                with tab_sell:
                    if comp.sell_signals:
                        for s in comp.sell_signals:
                            st.markdown(f"- ⚠️ {s}")
                    else:
                        st.markdown("*売りシグナルなし*")
                with tab_neutral:
                    for s in comp.neutral_signals:
                        st.markdown(f"- ➡️ {s}")

        st.markdown("---")

        # ── RSI ──────────────────────────────────
        col_rsi_chart, col_rsi_info = st.columns([3, 1])
        with col_rsi_chart:
            st.markdown("**RSI（14）**　— 相対力指数")
            rsi_fig = create_rsi_chart(ind, result.df)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with col_rsi_info:
            if rsi_r:
                st.markdown("&nbsp;")
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>RSI値</h4>
                    <div class="value" style="color:{rsi_r.color};">{rsi_r.value:.1f}</div>
                    <div style="color:#8899AA; font-size:12px; margin-top:4px;">{rsi_r.emoji} {rsi_r.signal}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
                if rsi_r.divergence != "なし":
                    st.warning(f"⚡ {rsi_r.divergence}")

        # ── MACD ─────────────────────────────────
        col_macd_chart, col_macd_info = st.columns([3, 1])
        with col_macd_chart:
            st.markdown("**MACD（12, 26, 9）**　— モメンタム")
            macd_fig = create_macd_chart(ind, result.df)
            st.plotly_chart(macd_fig, use_container_width=True)

        with col_macd_info:
            if macd_r:
                cross_color = "#00CC66" if macd_r.cross_signal == "ゴールデンクロス" else \
                              "#FF4444" if macd_r.cross_signal == "デッドクロス" else "#AAAAAA"
                trend_color = "#00CC66" if macd_r.trend == "上昇" else "#FF4444"
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>MACDシグナル</h4>
                    <div style="color:{cross_color}; font-weight:bold; font-size:13px;">
                      {macd_r.cross_signal if macd_r.cross_signal != 'なし' else 'クロスなし'}
                    </div>
                    <div style="color:{trend_color}; font-size:12px; margin-top:4px;">
                      トレンド: {macd_r.trend}
                    </div>
                    <div style="color:#8899AA; font-size:11px; margin-top:4px;">
                      {macd_r.histogram_trend}
                    </div>
                    </div>""",
                    unsafe_allow_html=True
                )

        # ── OBV + ボリューム ───────────────────────
        col_obv_chart, col_obv_info = st.columns([3, 1])
        with col_obv_chart:
            st.markdown("**OBV（累積出来高）**　— Smart Money追跡")
            obv_fig = create_obv_chart(ind, result.df)
            st.plotly_chart(obv_fig, use_container_width=True)

        with col_obv_info:
            if vol_r:
                obv_color = "#00CC66" if vol_r.obv_trend == "上昇" else \
                            "#FF4444" if vol_r.obv_trend == "下落" else "#AAAAAA"
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>出来高分析</h4>
                    <div style="color:{obv_color}; font-weight:bold;">OBV: {vol_r.obv_trend}</div>
                    <div style="color:#8899AA; font-size:12px; margin-top:4px;">
                      出来高倍率: {vol_r.vol_ratio:.1f}x
                    </div>
                    <div style="color:#8899AA; font-size:12px;">
                      トレンド: {vol_r.volume_trend}
                    </div>
                    </div>""",
                    unsafe_allow_html=True
                )
                if vol_r.obv_divergence != "なし":
                    div_color = "#00CC66" if "強気" in vol_r.obv_divergence else "#FF8800"
                    st.markdown(
                        f'<div style="color:{div_color}; font-size:12px; padding:6px;">⚡ OBV {vol_r.obv_divergence}</div>',
                        unsafe_allow_html=True
                    )
                if vol_r.is_high_volume:
                    st.success(f"🔥 出来高急増！ ({vol_r.vol_ratio:.1f}x 平均比)")

        st.markdown("---")

        # ── 4列グリッド: BB / Ichimoku / Momentum / Cross ──────
        col_bb, col_ichi, col_mom, col_cross = st.columns(4)

        with col_bb:
            st.markdown("**ボリンジャーバンド**")
            if bb_r:
                pct_color = "#00CC66" if bb_r.pct_b < 0.2 else \
                            "#FF4444" if bb_r.pct_b > 0.8 else "#AAAAAA"
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>%B（バンド内位置）</h4>
                    <div class="value" style="color:{pct_color};">{bb_r.pct_b:.2f}</div>
                    <div style="color:#8899AA; font-size:11px; margin-top:4px;">
                      0=下限 | 0.5=中央 | 1=上限
                    </div>
                    </div>""",
                    unsafe_allow_html=True
                )
                if bb_r.squeeze:
                    st.warning("⚡ BBスクイーズ中（大きな動き準備）")
                if bb_r.squeeze_release:
                    st.success("🚀 スクイーズ解放！ブレイクアウト")
                if bb_r.walk_upper:
                    st.info("📈 上バンドウォーク（強いトレンド）")
                if bb_r.walk_lower:
                    st.error("📉 下バンドウォーク（強い下落）")
                st.markdown(f"""
                - 上限: `{bb_r.upper:.2f}`
                - 中心: `{bb_r.middle:.2f}`
                - 下限: `{bb_r.lower:.2f}`
                - バンド幅: `{bb_r.width:.3f}`
                """)

        with col_ichi:
            st.markdown("**一目均衡表**")
            if ichi_r:
                cloud_color_label = "🟢 緑雲（強気）" if ichi_r.cloud_color == "bullish" else \
                                    "🔴 赤雲（弱気）" if ichi_r.cloud_color == "bearish" else "⚪ 中立"
                pos_label = "☁️ 雲の上（強気）" if ichi_r.above_cloud else \
                            "🌩️ 雲の下（弱気）" if ichi_r.below_cloud else "☁️ 雲の中（中立）"
                pos_color = "#00CC66" if ichi_r.above_cloud else \
                            "#FF4444" if ichi_r.below_cloud else "#AAAAAA"
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>株価位置</h4>
                    <div style="color:{pos_color}; font-weight:bold; font-size:13px;">{pos_label}</div>
                    <div style="color:#8899AA; font-size:11px; margin-top:4px;">{cloud_color_label}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
                if ichi_r.tk_cross != "なし":
                    tk_color = "#00CC66" if "ゴールデン" in ichi_r.tk_cross else "#FF4444"
                    st.markdown(f'<div style="color:{tk_color}; font-size:12px;">⚡ TKクロス: {ichi_r.tk_cross}</div>',
                               unsafe_allow_html=True)
                st.markdown(f"""
                - 転換線: `{ichi_r.tenkan:.2f}`
                - 基準線: `{ichi_r.kijun:.2f}`
                - 先行A: `{ichi_r.senkou_a:.2f}`
                - 先行B: `{ichi_r.senkou_b:.2f}`
                """)

        with col_mom:
            st.markdown("**モメンタム / ROC**")
            if mom_r:
                mom_color = "#00CC66" if mom_r.momentum_score > 0.3 else \
                            "#FF4444" if mom_r.momentum_score < -0.3 else "#AAAAAA"
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>総合モメンタム</h4>
                    <div class="value" style="color:{mom_color};">{mom_r.momentum_score:+.2f}</div>
                    <div style="color:#8899AA; font-size:11px; margin-top:4px;">{mom_r.rank_signal}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
                def roc_color(v):
                    return "#00CC66" if v > 0 else "#FF4444"
                st.markdown(f"""
                | 期間 | リターン |
                |------|---------|
                | 1ヶ月 | <span style="color:{roc_color(mom_r.roc_1m)}">{mom_r.roc_1m:+.1f}%</span> |
                | 3ヶ月 | <span style="color:{roc_color(mom_r.roc_3m)}">{mom_r.roc_3m:+.1f}%</span> |
                | 6ヶ月 | <span style="color:{roc_color(mom_r.roc_6m)}">{mom_r.roc_6m:+.1f}%</span> |
                | 12ヶ月 | <span style="color:{roc_color(mom_r.roc_12m)}">{mom_r.roc_12m:+.1f}%</span> |
                """, unsafe_allow_html=True)

        with col_cross:
            st.markdown("**ゴールデン/デッドクロス**")
            if cross_r:
                if cross_r.golden_cross_50_200:
                    st.success("🌟 50MA/200MAゴールデンクロス")
                elif cross_r.death_cross_50_200:
                    st.error("💀 50MA/200MAデッドクロス")
                elif cross_r.days_since_cross > 0:
                    st.info(f"⚡ クロス後 {cross_r.days_since_cross}日経過")
                else:
                    st.markdown("*クロスなし（継続中）*")

                if cross_r.broke_52w_high:
                    st.success("🚀 52週高値ブレイク！（実証済みアノマリー）")
                elif cross_r.near_52w_high:
                    st.warning("👀 52週高値に接近（ブレイク待ち）")

        st.markdown("---")

        # ── ATR（ボラティリティ・リスク管理） ────────────
        col_atr_info, col_atr_chart = st.columns([1, 2])
        with col_atr_info:
            st.markdown("**ATR（Average True Range）**")
            st.markdown("*ポジションサイズと損切り幅の定量化*")
            if atr_r:
                vol_color = "#00CC66" if atr_r.volatility_regime == "低" else \
                            "#FF4444" if atr_r.volatility_regime == "高" else "#FFB800"
                is_jp = ".T" in result.ticker
                currency = "¥" if is_jp else "$"
                st.markdown(
                    f"""<div class="metric-card">
                    <h4>ATR(14)</h4>
                    <div class="value">{currency}{atr_r.atr_14:.2f}</div>
                    <div style="color:{vol_color}; font-size:12px; margin-top:4px;">
                      ボラティリティ: {atr_r.volatility_regime} ({atr_r.atr_pct:.1f}%)
                    </div>
                    </div>""",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                | 用途 | 価格 |
                |------|------|
                | 推奨損切り（×2） | `{currency}{atr_r.suggested_stop_atr:.2f}` |
                | 1:3目標（×6） | `{currency}{atr_r.suggested_target_atr:.2f}` |
                """)

        with col_atr_chart:
            if atr_r:
                atr_fig = create_atr_range_chart(atr_r, result.current_price, ".T" in result.ticker)
                st.plotly_chart(atr_fig, use_container_width=True)

        # ── フィボナッチ ──────────────────────────────
        if fib_r and fib_r.levels:
            with st.expander("📐 フィボナッチ・リトレースメント"):
                st.markdown(f"""
                **方向**: {fib_r.direction}トレンドを基準に計算
                **スイングハイ**: `{fib_r.swing_high:.2f}` / **スイングロー**: `{fib_r.swing_low:.2f}`
                """)
                fib_labels = {
                    0.236: "23.6%",
                    0.382: "38.2%",
                    0.500: "50.0%",
                    0.618: "61.8% ★（黄金比）",
                    0.786: "78.6%",
                }
                fib_rows = []
                current_price = result.current_price
                for ratio in sorted(fib_r.levels.keys(), reverse=True):
                    price = fib_r.levels[ratio]
                    dist = (price - current_price) / current_price * 100
                    marker = "← 現在値" if abs(dist) < 1 else ""
                    is_nearest = (
                        (fib_r.nearest_resistance is not None and abs(price - fib_r.nearest_resistance) < 0.01) or
                        (fib_r.nearest_support is not None and abs(price - fib_r.nearest_support) < 0.01)
                    )
                    fib_rows.append({
                        "比率": fib_labels.get(ratio, f"{ratio:.1%}"),
                        "価格": f"{price:.2f}",
                        "現在値比": f"{dist:+.1f}%",
                        "注目": "⭐ 最近" if is_nearest else ""
                    })
                st.dataframe(pd.DataFrame(fib_rows), use_container_width=True, hide_index=True)

    # ── ギャップ一覧 ──────────────────────────────
    if show_gaps and result.gaps:
        st.markdown("---")
        with st.expander(f"🕳️ ギャップ（窓）一覧 — {len(result.gaps)}個検出"):
            gap_data = []
            for g in result.gaps[-10:]:  # 直近10個
                status = "✅ 埋まり済" if g.is_filled else "🔲 未埋め"
                gap_type = "↑ ギャップアップ" if g.gap_type == "up" else "↓ ギャップダウン"
                gap_data.append({
                    "日付": str(g.date.date()),
                    "種類": gap_type,
                    "上端": f"{g.upper:.2f}",
                    "下端": f"{g.lower:.2f}",
                    "幅": f"{(g.upper - g.lower) / g.lower * 100:.1f}%",
                    "状態": status
                })
            st.dataframe(pd.DataFrame(gap_data), use_container_width=True, hide_index=True)

    # ── フッター ──────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#445566; font-size:11px; padding:10px;">
    Y式「チャートの詠み方 極」に基づく分析ツール<br>
    ⚠️ 本ツールは投資判断の補助ツールです。最終判断は自己責任でお願いします。
    </div>
    """, unsafe_allow_html=True)
