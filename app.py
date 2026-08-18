import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(
    page_title='Chart Signal Analyzer',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

# iPad / mobile friendly spacing and larger touch targets.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      div[data-baseweb="select"] > div, .stTextInput input {min-height: 44px;}
      .stButton button {min-height: 44px; font-size: 1rem;}
      @media (max-width: 900px) {
        .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
        h1 {font-size: 1.8rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class SignalResult:
    score: int
    label: str
    reasons: list[str]
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out['Close']
    out['EMA20'] = close.ewm(span=20, adjust=False).mean()
    out['EMA50'] = close.ewm(span=50, adjust=False).mean()
    out['EMA200'] = close.ewm(span=200, adjust=False).mean()
    out['RSI14'] = rsi(close, 14)
    out['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out['MACD_SIGNAL'] = out['MACD'].ewm(span=9, adjust=False).mean()
    out['MACD_HIST'] = out['MACD'] - out['MACD_SIGNAL']
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out['BB_MID'] = mid
    out['BB_UPPER'] = mid + 2 * std
    out['BB_LOWER'] = mid - 2 * std
    tr = pd.concat([
        out['High'] - out['Low'],
        (out['High'] - out['Close'].shift()).abs(),
        (out['Low'] - out['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    out['ATR14'] = tr.rolling(14).mean()
    out['VOL_MA20'] = out['Volume'].rolling(20).mean()
    return out


def score_signal(df: pd.DataFrame) -> SignalResult:
    row = df.iloc[-1]
    prev = df.iloc[-2]
    score = 50
    reasons: list[str] = []

    if row['Close'] > row['EMA20'] > row['EMA50']:
        score += 14
        reasons.append('Kurzfristiger Trend bullish: Kurs > EMA20 > EMA50')
    elif row['Close'] < row['EMA20'] < row['EMA50']:
        score -= 14
        reasons.append('Kurzfristiger Trend bearish: Kurs < EMA20 < EMA50')

    if not math.isnan(row['EMA200']):
        if row['Close'] > row['EMA200']:
            score += 8
            reasons.append('Kurs liegt über EMA200')
        else:
            score -= 8
            reasons.append('Kurs liegt unter EMA200')

    if 45 <= row['RSI14'] <= 65:
        score += 8
        reasons.append(f'RSI {row["RSI14"]:.1f}: konstruktives Momentum')
    elif row['RSI14'] < 30:
        score += 5
        reasons.append(f'RSI {row["RSI14"]:.1f}: überverkauft, Rebound möglich')
    elif row['RSI14'] > 70:
        score -= 8
        reasons.append(f'RSI {row["RSI14"]:.1f}: überkauft')

    macd_cross_up = prev['MACD'] <= prev['MACD_SIGNAL'] and row['MACD'] > row['MACD_SIGNAL']
    macd_cross_down = prev['MACD'] >= prev['MACD_SIGNAL'] and row['MACD'] < row['MACD_SIGNAL']
    if macd_cross_up:
        score += 10
        reasons.append('MACD frisches Kaufsignal')
    elif macd_cross_down:
        score -= 10
        reasons.append('MACD frisches Verkaufssignal')
    elif row['MACD'] > row['MACD_SIGNAL']:
        score += 5
        reasons.append('MACD positiv')
    else:
        score -= 5
        reasons.append('MACD negativ')

    if row['Volume'] > row['VOL_MA20'] * 1.15:
        if row['Close'] > prev['Close']:
            score += 7
            reasons.append('Steigender Kurs mit überdurchschnittlichem Volumen')
        else:
            score -= 7
            reasons.append('Fallender Kurs mit überdurchschnittlichem Volumen')

    if row['Close'] < row['BB_LOWER']:
        score += 5
        reasons.append('Kurs unter unterem Bollinger-Band')
    elif row['Close'] > row['BB_UPPER']:
        score -= 5
        reasons.append('Kurs über oberem Bollinger-Band')

    score = int(max(0, min(100, round(score))))
    if score >= 70:
        label = 'KAUF'
    elif score <= 30:
        label = 'VERKAUF'
    else:
        label = 'NEUTRAL / ABWARTEN'

    entry = float(row['Close'])
    atr = float(row['ATR14']) if pd.notna(row['ATR14']) and row['ATR14'] > 0 else entry * 0.02
    stop = max(0.01, entry - 1.5 * atr)
    risk = entry - stop
    tp1 = entry + 2 * risk
    tp2 = entry + 3 * risk
    return SignalResult(score, label, reasons, entry, stop, tp1, tp2)


@st.cache_data(ttl=300)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how='all')


st.title('📈 Chart Signal Analyzer')
st.caption('Technische Analyse mit Kauf-/Verkaufsscore. Kein automatischer Handel und keine Anlageberatung.')

with st.sidebar:
    st.header('Analyse')

    markets = {
        'Ethereum (ETH/USD)': 'ETH-USD',
        'Bitcoin (BTC/USD)': 'BTC-USD',
        'DAX 40': '^GDAXI',
        'US 100 / Nasdaq 100': '^NDX',
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'Gold': 'GC=F',
        'Silber': 'SI=F',
        'Nvidia': 'NVDA',
        'Apple': 'AAPL',
        'Tesla': 'TSLA',
        'Eigener Ticker': '',
    }
    market = st.selectbox('Markt', list(markets.keys()), index=0)
    if market == 'Eigener Ticker':
        ticker = st.text_input('Ticker', value='ETH-USD').strip().upper()
    else:
        ticker = markets[market]
        st.caption(f'Yahoo-Finance-Symbol: {ticker}')

    period = st.selectbox('Zeitraum', ['1mo', '3mo', '6mo', '1y', '2y'], index=2)
    interval = st.selectbox('Intervall', ['15m', '30m', '1h', '1d'], index=2)
    st.info('Enthalten: DAX 40, US 100, S&P 500, Dow Jones, Gold, Silber, BTC, ETH und Aktien.')

try:
    raw = load_data(ticker, period, interval)
    if raw.empty or len(raw) < 60:
        st.error('Zu wenige Kursdaten. Wähle einen längeren Zeitraum oder ein größeres Intervall.')
        st.stop()
    df = add_indicators(raw)
    result = score_signal(df)
    last = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Signal', result.label)
    c2.metric('Score', f'{result.score}/100')
    c3.metric('Kurs', f'{result.entry:,.2f}')
    c4.metric('RSI 14', f'{last["RSI14"]:.1f}')

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Kurs',
        ),
        row=1,
        col=1,
    )
    for col, name in [('EMA20', 'EMA20'), ('EMA50', 'EMA50'), ('EMA200', 'EMA200')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], mode='lines', name='BB oben', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], mode='lines', name='BB unten', line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI14'), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], mode='lines', name='Signal'), row=3, col=1)
    fig.update_layout(
        height=820,
        xaxis_rangeslider_visible=False,
        legend_orientation='h',
        margin=dict(l=10, r=10, t=35, b=10),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False, 'responsive': True})

    st.subheader('Signalbewertung')
    left, right = st.columns([1, 1])
    with left:
        for reason in result.reasons:
            st.write(f'• {reason}')
    with right:
        levels = pd.DataFrame({
            'Level': ['Einstieg', 'Stop-Loss', 'Take-Profit 1', 'Take-Profit 2'],
            'Kurs': [result.entry, result.stop_loss, result.take_profit_1, result.take_profit_2],
        })
        st.dataframe(levels, hide_index=True, use_container_width=True)

    st.caption('Der Score ist regelbasiert und dient als Analysehilfe. Vor echtem Einsatz sollte die Strategie per Backtest und Paper-Trading geprüft werden.')
except Exception as exc:
    st.error(f'Daten/Analyse konnten nicht geladen werden: {exc}')
