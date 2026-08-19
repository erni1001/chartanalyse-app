import json
import math
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title='Chart Signal Analyzer', page_icon='📈', layout='wide')


PRESETS = {
    'Ethereum (ETH/USD)': {'td': 'ETH/USD', 'yf': 'ETH-USD'},
    'Bitcoin (BTC/USD)': {'td': 'BTC/USD', 'yf': 'BTC-USD'},
    'DAX 40': {'td': 'DAX', 'yf': '^GDAXI'},
    'US 100 / Nasdaq 100': {'td': 'NDX', 'yf': '^NDX'},
    'S&P 500': {'td': 'SPX', 'yf': '^GSPC'},
    'Dow Jones': {'td': 'DJI', 'yf': '^DJI'},
    'Gold Spot': {'td': 'XAU/USD', 'yf': 'GC=F'},
    'Silber Spot': {'td': 'XAG/USD', 'yf': 'SI=F'},
    'Nvidia': {'td': 'NVDA', 'yf': 'NVDA'},
    'Apple': {'td': 'AAPL', 'yf': 'AAPL'},
    'Tesla': {'td': 'TSLA', 'yf': 'TSLA'},
    'Eigener Ticker': {'td': '', 'yf': ''},
}

INTERVAL_MAP = {
    '5m': ('5min', '5m'),
    '15m': ('15min', '15m'),
    '30m': ('30min', '30m'),
    '1h': ('1h', '1h'),
    '4h': ('4h', '1h'),
    '1D': ('1day', '1d'),
}

OUTPUTSIZE_MAP = {
    '1 Monat': 250,
    '3 Monate': 500,
    '6 Monate': 1000,
    '1 Jahr': 1500,
}

YF_PERIOD_MAP = {
    '1 Monat': '1mo',
    '3 Monate': '3mo',
    '6 Monate': '6mo',
    '1 Jahr': '1y',
}


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

    if pd.notna(row['EMA200']):
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

    if row['Volume'] > 0 and pd.notna(row['VOL_MA20']) and row['VOL_MA20'] > 0:
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
        label = 'KAUF / LONG'
    elif score <= 30:
        label = 'VERKAUF / SHORT'
    else:
        label = 'ABWARTEN'

    entry = float(row['Close'])
    atr = float(row['ATR14']) if pd.notna(row['ATR14']) and row['ATR14'] > 0 else entry * 0.02
    stop = max(0.01, entry - 1.5 * atr)
    risk = entry - stop
    tp1 = entry + 2 * risk
    tp2 = entry + 3 * risk
    return SignalResult(score, label, reasons, entry, stop, tp1, tp2)


def _twelve_request(params: dict) -> dict:
    api_key = st.secrets.get('TWELVE_DATA_API_KEY', '')
    if not api_key:
        raise RuntimeError('TWELVE_DATA_API_KEY ist in Streamlit Secrets nicht gesetzt.')
    params = {**params, 'apikey': api_key}
    url = 'https://api.twelvedata.com/time_series?' + urlencode(params)
    req = Request(url, headers={'User-Agent': 'ChartSignalAnalyzer/1.0'})
    with urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode('utf-8'))


@st.cache_data(ttl=120)
def load_twelve_data(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    payload = _twelve_request({
        'symbol': symbol,
        'interval': interval,
        'outputsize': min(int(outputsize), 5000),
        'format': 'JSON',
        'order': 'asc',
        'timezone': 'UTC',
    })
    if payload.get('status') == 'error':
        raise RuntimeError(payload.get('message', 'Unbekannter Twelve-Data-Fehler'))
    values = payload.get('values') or []
    if not values:
        raise RuntimeError('Twelve Data hat keine Kursdaten geliefert.')

    df = pd.DataFrame(values)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.set_index('datetime').sort_index()
    rename = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df = df.rename(columns=rename)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Volume' not in df:
        df['Volume'] = 0.0
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0.0)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Open', 'High', 'Low', 'Close'])


@st.cache_data(ttl=300)
def load_yahoo_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # 4h gibt es bei yfinance nicht direkt; 1h laden und auf 4h resamplen.
    yf_interval = '1h' if interval == '4h' else interval
    df = yf.download(ticker, period=period, interval=yf_interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how='all')
    if interval == '4h' and not df.empty:
        df = df.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
    return df


def format_price(value: float) -> str:
    if value >= 1000:
        return f'{value:,.2f}'
    if value >= 1:
        return f'{value:,.4f}'
    return f'{value:,.6f}'


st.title('📈 Chart Signal Analyzer')
st.caption('Twelve Data als Hauptquelle, Yahoo Finance als Fallback. Technische Analyse, kein automatischer Handel.')

with st.sidebar:
    st.header('Analyse')
    preset = st.selectbox('Markt', list(PRESETS.keys()), index=0)
    if preset == 'Eigener Ticker':
        td_symbol = st.text_input('Twelve-Data-Symbol', value='AAPL').strip().upper()
        yf_symbol = st.text_input('Yahoo-Fallback-Symbol', value='AAPL').strip().upper()
    else:
        td_symbol = PRESETS[preset]['td']
        yf_symbol = PRESETS[preset]['yf']
        st.caption(f'Twelve Data: {td_symbol} · Yahoo Fallback: {yf_symbol}')

    period_label = st.selectbox('Zeitraum', list(OUTPUTSIZE_MAP.keys()), index=2)
    interval_label = st.selectbox('Intervall', list(INTERVAL_MAP.keys()), index=3)
    source_mode = st.selectbox('Datenquelle', ['Twelve Data → Yahoo Fallback', 'Nur Twelve Data', 'Nur Yahoo Finance'])

    st.info('Twelve Data unterstützt u. a. 5m, 15m, 30m, 1h, 4h und 1day. Der kostenlose Plan kann je nach Symbol/Markt Einschränkungen haben.')

try:
    td_interval, yf_interval = INTERVAL_MAP[interval_label]
    outputsize = OUTPUTSIZE_MAP[period_label]
    yf_period = YF_PERIOD_MAP[period_label]

    raw = pd.DataFrame()
    source_used = ''
    td_error = None

    if source_mode != 'Nur Yahoo Finance':
        try:
            raw = load_twelve_data(td_symbol, td_interval, outputsize)
            source_used = f'Twelve Data · {td_symbol} · {td_interval}'
        except Exception as exc:
            td_error = str(exc)
            if source_mode == 'Nur Twelve Data':
                raise

    if raw.empty and source_mode != 'Nur Twelve Data':
        raw = load_yahoo_data(yf_symbol, yf_period, yf_interval)
        source_used = f'Yahoo Finance Fallback · {yf_symbol} · {yf_interval}'

    if raw.empty or len(raw) < 60:
        st.error('Zu wenige Kursdaten. Wähle einen längeren Zeitraum oder ein größeres Intervall.')
        if td_error:
            st.caption(f'Twelve-Data-Hinweis: {td_error}')
        st.stop()

    df = add_indicators(raw)
    result = score_signal(df)
    last = df.iloc[-1]

    st.caption(f'Datenquelle: **{source_used}** · letzter Datenpunkt: **{df.index[-1]}**')
    if td_error and source_used.startswith('Yahoo'):
        st.warning(f'Twelve Data konnte für dieses Symbol/Intervall nicht genutzt werden; Yahoo-Fallback aktiv. Grund: {td_error}')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Signal', result.label)
    c2.metric('Score', f'{result.score}/100')
    c3.metric('Kurs', format_price(result.entry))
    c4.metric('RSI 14', f'{last["RSI14"]:.1f}')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.58, 0.20, 0.22])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Kurs'), row=1, col=1)
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
    fig.update_layout(height=850, xaxis_rangeslider_visible=False, legend_orientation='h')
    st.plotly_chart(fig, width='stretch')

    st.subheader('Signalbewertung')
    left, right = st.columns([1, 1])
    with left:
        for reason in result.reasons:
            st.write(f'• {reason}')
    with right:
        risk = result.entry - result.stop_loss
        levels = pd.DataFrame({
            'Level': ['Einstieg', 'Stop-Loss', 'Take-Profit 1', 'Take-Profit 2'],
            'Kurs': [result.entry, result.stop_loss, result.take_profit_1, result.take_profit_2],
            'CRV': ['', '1.0 R', '2.0 R', '3.0 R'],
        })
        st.dataframe(levels, hide_index=True, width='stretch')

    st.caption('Der Score ist regelbasiert und keine garantierte Trefferwahrscheinlichkeit. Vor echtem Einsatz per Backtest und Paper-Trading prüfen.')
except Exception as exc:
    st.error(f'Daten/Analyse konnten nicht geladen werden: {exc}')
