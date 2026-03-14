import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
import webbrowser

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'docs'), exist_ok=True)

INDEX_TICKER = '^GSPC'
INDEX_START_DATE = '1990-01-01'
LEVERAGE_TICKER = 'UPRO'
TARGET_ETFS = ['SPY', 'VOO', 'IVV', 'SPYM', 'SSO']
HTML_PATH = os.path.join(BASE_DIR, 'docs', 'sp500_strategy.html')

TICKER_MERGE_MAP = {
    'SPYM': ['SPLG', 'SPYM']
}

RSI_PERIOD = 10  # ★ 固定RSI(10)

print(f"📂 数据存储目录: {DATA_DIR}")

# ============================================================
# 数据处理
# ============================================================
class CSVDataManager:
    def get_path(self, ticker):
        return os.path.join(DATA_DIR, f"{ticker.replace('^', '')}_daily.csv")

    def load_data(self, ticker):
        path = self.get_path(ticker)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col='date', parse_dates=True)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                return df
            except: return None
        return None

    def save_data(self, df, ticker):
        path = self.get_path(ticker)
        df_save = df.reset_index()
        df_save.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        if 'date' in df_save.columns:
            df_save['date'] = pd.to_datetime(df_save['date']).dt.strftime('%Y-%m-%d')
        df_save.to_csv(path, index=False, encoding='utf-8-sig')

def calculate_indicators(df):
    """计算 MA200 和 RSI(10)"""
    data = df.copy().sort_index()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    period = RSI_PERIOD
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    data['RSI_10'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    return data

def get_usd_hkd_rate():
    try:
        fx = yf.Ticker("HKD=X")
        hist = fx.history(period="5d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return 7.8
    except: return 7.8

def fetch_new_data(ticker, start_date=None):
    print(f"  ⬇️ 下载 {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date) if start_date else stock.history(period="max")
        if data.empty: return pd.DataFrame()
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"  ⚠️ 下载 {ticker} 失败: {e}")
        return pd.DataFrame()

def update_ticker_data(dm, ticker, earliest_date=None):
    old_data = dm.load_data(ticker)
    if old_data is not None and not old_data.empty:
        start_fetch = old_data.index[-1] - timedelta(days=7)
        new_data = fetch_new_data(ticker, start_date=start_fetch)
        full_data = pd.concat([old_data[['Open', 'High', 'Low', 'Close', 'Volume']], new_data])
        full_data = full_data[~full_data.index.duplicated(keep='last')].sort_index()
    else:
        full_data = fetch_new_data(ticker, start_date=earliest_date)
    dm.save_data(full_data, ticker)
    return full_data

def update_merged_ticker(dm, target_ticker, source_tickers):
    print(f"📦 合并下载 {source_tickers} → {target_ticker}")
    existing = dm.load_data(target_ticker)
    all_frames = []
    if existing is not None and not existing.empty:
        all_frames.append(existing[['Open', 'High', 'Low', 'Close', 'Volume']])
    for src in source_tickers:
        if existing is not None and not existing.empty:
            start_fetch = existing.index[-1] - timedelta(days=7)
            data = fetch_new_data(src, start_date=start_fetch)
        else:
            data = fetch_new_data(src)
        if not data.empty:
            all_frames.append(data)
    if not all_frames:
        print(f"  ⚠️ {target_ticker} 无数据")
        return pd.DataFrame()
    full_data = pd.concat(all_frames)
    full_data = full_data[~full_data.index.duplicated(keep='last')].sort_index()
    dm.save_data(full_data, target_ticker)
    date_range = f"{full_data.index[0].strftime('%Y-%m-%d')} ~ {full_data.index[-1].strftime('%Y-%m-%d')}"
    print(f"  ✅ {target_ticker} 合并完成: {len(full_data)} 条 ({date_range})")
    return full_data

def prepare_json_data(index_df, etf_data_map, upro_df):
    print("📦 正在对齐所有标的数据...")
    df = pd.DataFrame(index=index_df.index)
    df['date'] = df.index.strftime('%Y-%m-%d')
    df['Index_Close'] = index_df['Close'].round(2)
    df['Index_MA200'] = index_df['MA200'].round(2)
    df['RSI_10'] = index_df['RSI_10'].round(2)
    df['UPRO'] = upro_df['Close'].reindex(df.index).round(3)
    for ticker, data in etf_data_map.items():
        key = ticker.replace('^', '')
        df[key] = data['Close'].reindex(df.index).round(3)
    df = df[df.index >= '1990-01-01']
    df = df.dropna(subset=['Index_Close'])
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient='records')

# ============================================================
# HTML 生成
# ============================================================
def generate_interactive_html(data_json, usd_hkd):
    json_str = json.dumps(data_json)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&P 500 定投及增益策略</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #1a1a2e; --card-bg: #232336; --accent: #00d4ff;
            --bull: #00ff88; --bear: #ff6b6b; --text: #e0e0e0;
            --input-bg: #2a2a40; --input-border: #444; --warn: #ffaa00;
        }}
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:'Segoe UI',Roboto,sans-serif;background-color:var(--bg);color:var(--text);padding:20px;font-size:14px;}}
        .container{{max-width:1400px;margin:0 auto}}

        .controls {{
            display: flex; flex-wrap: wrap; gap: 15px;
            background: rgba(255,255,255,0.05);
            padding: 20px; border-radius: 12px; margin-bottom: 20px;
            align-items: flex-end; border: 1px solid rgba(255,255,255,0.1);
        }}
        .control-group {{ display: flex; flex-direction: column; gap: 6px; }}
        .control-group label {{ font-size: 0.9em; color: #aaa; font-weight: 500; }}

        .checkbox-group {{
            display: flex; align-items: center; gap: 15px;
            background: var(--input-bg); padding: 8px 12px;
            border: 1px solid var(--input-border); border-radius: 6px; height: 38px;
        }}
        .checkbox-group label {{ cursor: pointer; color: #fff; display: flex; align-items: center; gap: 5px; margin:0;}}
        input[type="checkbox"] {{ accent-color: var(--accent); width: 16px; height: 16px; margin: 0; min-width: auto; }}

        select, input[type="number"], input[type="date"] {{
            background-color: var(--input-bg); color: #fff; border: 1px solid var(--input-border);
            padding: 8px 12px; border-radius: 6px; outline: none; font-size: 14px; min-width: 90px;
        }}
        select:focus, input:focus {{ border-color: var(--accent); }}

        button {{
            background: linear-gradient(135deg, #00d4ff, #0077ff); color: #fff; border: none;
            padding: 9px 25px; border-radius: 6px; font-weight: bold; cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,119,255,0.3); transition: 0.2s;
        }}
        button:hover {{ opacity: 0.9; transform: translateY(-2px); }}

        .strategy-selector {{
            display: flex; align-items: center; gap: 10px;
            background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,119,255,0.08));
            padding: 12px 20px; border-radius: 12px; margin-bottom: 15px;
            border: 1px solid rgba(0,212,255,0.3);
        }}
        .strategy-selector label {{ font-weight: bold; color: var(--accent); font-size: 1.1em; }}
        .strategy-selector select {{ font-size: 1.05em; padding: 8px 16px; font-weight: bold; }}

        .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:20px}}
        .card{{background:var(--card-bg);border-radius:12px;padding:15px;border:1px solid rgba(255,255,255,0.05); box-shadow: 0 4px 6px rgba(0,0,0,0.1);}}
        .card h3{{color:#888;font-size:0.85em;margin-bottom:8px;text-transform:uppercase;}}
        .card .val{{font-size:1.5em;font-weight:bold;color:#fff}}
        .card .sub{{font-size:0.8em;color:#888;margin-top:4px}}

        .action-box{{background:linear-gradient(135deg, rgba(0,212,255,0.1), rgba(0,212,255,0.02));border:1px solid rgba(0,212,255,0.3);border-radius:12px;padding:15px;text-align:center;margin-bottom:20px; display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap;}}
        .action-item {{ margin: 10px; }}
        .action-val {{ font-size: 1.8em; font-weight: bold; margin: 5px 0; }}

        .pk-box {{
            background: linear-gradient(135deg, rgba(255,170,0,0.08), rgba(255,170,0,0.02));
            border: 1px solid rgba(255,170,0,0.3); border-radius: 12px; padding: 15px;
            text-align: center; margin-bottom: 20px;
            display: none; justify-content: space-around; align-items: center; flex-wrap: wrap;
        }}
        .pk-box.active {{ display: flex; }}
        .pk-item {{ margin: 10px; min-width: 150px; }}
        .pk-label {{ font-size: 0.85em; color: #aaa; margin-bottom: 4px; }}
        .pk-val {{ font-size: 1.4em; font-weight: bold; }}

        .chart-container {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 20px; }}
        .chart-box{{background:var(--card-bg);padding:15px;border-radius:12px; border:1px solid rgba(255,255,255,0.05);}}
        .chart-title {{ font-size: 1.1em; color: var(--accent); margin-bottom: 10px; font-weight: bold; }}

        .legend-bar {{
            display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;
            background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; margin-bottom: 10px;
            font-size: 0.9em; color: #ccc;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}

        table{{width:100%;border-collapse:collapse;font-size:0.9em}}
        th,td{{padding:10px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1)}}
        th{{background:rgba(0,0,0,0.2); position:sticky; top:0; color: #aaa;}}
        .scroll-table {{ max-height: 400px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: #444 #222; }}

        .c-bull {{ color: var(--bull); }} .c-bear {{ color: var(--bear); }} .c-warn {{ color: #ffaa00; }}
        tr.highlight {{ background: rgba(255, 170, 0, 0.1); }}
        .ipo-hint {{ font-size: 0.75em; color: #666; margin-top: 4px; line-height: 1.4; }}

        .pos-tag {{
            display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;
        }}
        .pos-main {{ background: rgba(0,255,136,0.15); color: var(--bull); }}
        .pos-upro100 {{ background: rgba(255,68,68,0.2); color: #ff4444; }}
        .pos-upro75 {{ background: rgba(255,136,0,0.2); color: #ff8800; }}
        .pos-upro50 {{ background: rgba(255,170,0,0.2); color: var(--warn); }}
        .pos-upro25 {{ background: rgba(255,204,0,0.2); color: #ffcc00; }}

        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div style="text-align:center; margin-bottom:20px;">
        <h2 style="color:var(--accent)">📈 S&P 500 定投及增益策略</h2>
        <div style="font-size:0.9em; color:#888">汇率: 1 USD = {usd_hkd:.4f} HKD | 分界线: 日线MA200 | RSI(10)</div>
    </div>

    <!-- ★ 策略选择器 -->
    <div class="strategy-selector">
        <label>🎯 投资策略:</label>
        <select id="strategyMode" onchange="onStrategyChange()">
            <option value="dca">📅 定投策略</option>
            <option value="bottom">🎯 抄底策略 </option>
        </select>
    </div>

    <div class="controls">
        <div class="control-group"><label>开始时间</label>
            <div style="display:flex; gap:5px;">
                <select id="periodSelect" style="width:110px" onchange="onPeriodChange()">
                    <option value="0">1993-2000</option>
                    <option value="1">2001-2010</option>
                    <option value="2">2011-2020</option>
                    <option value="3" selected>2021-至今</option>
                </select>
                <select id="startYear" style="width:70px" onchange="onDatePartChange()"></select>
                <select id="startMonth" style="width:60px" onchange="onDatePartChange()">
                    <option value="1">1月</option><option value="2">2月</option><option value="3">3月</option>
                    <option value="4">4月</option><option value="5">5月</option><option value="6">6月</option>
                    <option value="7">7月</option><option value="8">8月</option><option value="9">9月</option>
                    <option value="10">10月</option><option value="11">11月</option><option value="12">12月</option>
                </select>
                <select id="startDay" style="width:60px" onchange="onDatePartChange()"></select>
            </div>
        </div>

        <div class="control-group">
            <label style="color:var(--accent)">1. 投资标的①</label>
            <select id="tickerSelect" onchange="onTickerChange(1)">
                <option value="SPY">SPY (1993)</option>
                <option value="IVV">IVV (2000)</option>
                <option value="SPYM">SPYM (2005, 原SPLG)</option>
                <option value="VOO">VOO (2010)</option>
                <option value="SSO">SSO 2倍 (2006)</option>
                <option value="UPRO">UPRO 3倍 (2009)</option>
            </select>
            <div class="ipo-hint" id="ipoHint"></div>
        </div>

        <div class="control-group">
            <label style="color:#ffaa00">⚔️ 定投PK②</label>
            <select id="tickerSelect2" onchange="onTickerChange(2)">
                <option value="NONE">无 (不对比)</option>
                <option value="SPY">SPY (1993)</option>
                <option value="IVV">IVV (2000)</option>
                <option value="SPYM">SPYM (2005, 原SPLG)</option>
                <option value="VOO">VOO (2010)</option>
                <option value="SSO">SSO 2倍 (2006)</option>
                <option value="UPRO">UPRO 3倍 (2009)</option>
            </select>
            <div class="ipo-hint" id="ipoHint2"></div>
        </div>

        <div class="control-group">
            <label style="color:var(--bull)" id="lblStratConfig">3. 策略配置 (叠加)</label>
            <div class="checkbox-group">
                <label id="lblChkBase" title="每月固定日期定投基础金额"><input type="checkbox" id="chkBase" checked onchange="runStrategy()"> <span id="chkBaseText">基础定投</span></label>
                <label id="lblChkBear" title="当价格低于MA200时，当月定投金额翻倍"><input type="checkbox" id="chkBear" checked onchange="runStrategy()"> <span id="chkBearText">熊市加倍</span></label>
                <label id="lblChkExtreme" title="极端行情/标的切换"><input type="checkbox" id="chkExtreme" checked onchange="runStrategy()"> <span id="chkExtremeText">极端行情</span></label>
            </div>
        </div>

        <div class="control-group"><label>基准金额 (HKD)</label><input type="number" id="baseInvest" value="2000" step="500"></div>
        <div class="control-group"><label>查看截至</label><input type="date" id="endDate"></div>
        <div class="control-group"><label>&nbsp;</label><button onclick="runStrategy()">🔄 重新计算</button></div>
    </div>

    <div class="action-box">
        <div class="action-item"><div>当前持仓 (<span id="lblTicker1Tag">①</span>)</div><div id="statusPos" class="action-val" style="color:#fff">--</div><div id="statusPosDetail" style="font-size:0.75em; opacity:0.7">--</div></div>
        <div class="action-item"><div>下期建议 (<span id="lblTicker">SPY</span>)</div><div id="statusNextInvest" class="action-val">--</div><div id="statusMarket" style="font-size:0.8em; opacity:0.8">--</div></div>
        <div class="action-item"><div>总收益率 ①</div><div id="statusReturn" class="action-val">--</div></div>
        <div class="action-item"><div>年化收益率 ①</div><div id="statusAnnual" class="action-val">--</div><div id="statusAnnualNote" style="font-size:0.75em; opacity:0.7">N=<span id="annN">0</span>期 PMT=<span id="annPMT">0</span></div></div>
    </div>

    <div class="pk-box" id="pkBox">
        <div class="pk-item">
            <div class="pk-label">⚔️ 定投PK对比</div>
            <div class="pk-val" style="color:var(--accent)">① vs ②</div>
        </div>
        <div class="pk-item">
            <div class="pk-label" id="pkLabel1">① SPY</div>
            <div class="pk-val c-bull" id="pkReturn1">--</div>
            <div style="font-size:0.75em;color:#888">年化: <span id="pkAnnual1">--</span></div>
        </div>
        <div class="pk-item">
            <div class="pk-label" id="pkLabel2">② VOO</div>
            <div class="pk-val c-warn" id="pkReturn2">--</div>
            <div style="font-size:0.75em;color:#888">年化: <span id="pkAnnual2">--</span></div>
        </div>
        <div class="pk-item">
            <div class="pk-label">本金差</div>
            <div class="pk-val" id="pkInvDiff" style="color:#ccc">--</div>
        </div>
        <div class="pk-item">
            <div class="pk-label">市值差</div>
            <div class="pk-val" id="pkValDiff" style="color:#ccc">--</div>
        </div>
    </div>

    <div class="grid">
        <div class="card"><h3>标普500指数</h3><div class="val" id="valIdx">--</div><div class="sub" id="valDate">--</div></div>
        <div class="card"><h3>日线 MA200</h3><div class="val" id="valMa200">--</div><div class="sub">牛熊分界线</div></div>
        <div class="card"><h3>RSI (10)</h3><div class="val" id="valRsi">--</div><div class="sub" id="valRsiSig">--</div></div>
        <div class="card"><h3>份额详情 (①)</h3><div class="sub"><span id="lblShareTicker">SPY</span>: <span id="shareSpy" style="color:#fff">0</span></div><div class="sub">UPRO: <span id="shareUpro" style="color:var(--warn)">0</span></div><div class="sub" id="shareSpyRefuge" style="display:none">SPY(避险): <span id="shareSpyRef" style="color:#00d4ff">0</span></div></div>
        <div class="card"><h3>累计本金 ①</h3><div class="val" id="valTotalInv">--</div><div class="sub">HKD</div></div>
        <div class="card"><h3>总市值 ①</h3><div class="val" id="valTotalVal">--</div><div class="sub">HKD</div></div>
        <div class="card"><h3 id="lblCount1Title">基础定投期数</h3><div class="val" id="valBaseCount">0</div><div class="sub" id="lblCount1Sub">牛市单倍</div></div>
        <div class="card"><h3 id="lblCount2Title">熊市倍投期数</h3><div class="val" id="valBearCount">0</div><div class="sub" id="lblCount2Sub">熊市双倍</div></div>
    </div>

    <div class="chart-container">
        <div class="chart-box">
            <div class="chart-title">💰 账户总市值趋势 (HKD) <span id="chartPkHint" style="font-size:0.8em; color:#ffaa00"></span></div>
            <canvas id="mainChart" height="250" style="max-height:250px"></canvas>
        </div>
        <div class="chart-box">
            <div class="chart-title">📊 市场信号 (Index / MA200 / RSI)</div>
            <div class="legend-bar">
                <div class="legend-item"><span style="color:red">▲</span> 牛市RSI&lt;30切UPRO</div>
                <div style="color:#555">|</div>
                <div class="legend-item"><span style="color:blue">▼</span> &gt;70减25%</div>
                <div style="color:#555">|</div>
                <div class="legend-item"><span style="color:#00ff00">▼</span> &gt;80减25%</div>
                <div style="color:#555">|</div>
                <div class="legend-item"><span style="color:#ff00ff">▼</span> &gt;85减25%</div>
                <div style="color:#555">|</div>
                <div class="legend-item"><span style="color:#ff8800">▼</span> &lt;60减25%</div>
                <div style="color:#555">|</div>
                <div class="legend-item"><span style="color:#ff6b6b">━</span> 破MA200全清</div>
            </div>
            <canvas id="techChart" height="300" style="max-height:300px"></canvas>
        </div>
    </div>

    <div class="chart-box scroll-table">
        <div class="chart-title">📜 交易流水 ① (倒序)</div>
        <table id="logTable">
            <thead>
                <tr>
                    <th>日期</th><th>指数</th><th>RSI</th><th>牛熊</th>
                    <th>投入(HKD)</th><th>累计本金</th><th>总市值</th>
                    <th><span id="thShareTicker">SPY</span></th><th>UPRO</th><th>持仓</th><th>操作</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
</div>

<script>
const RAW_DATA = {json_str};
const USD_HKD = {usd_hkd};

const IPO_DATES = {{
    'SPY':  '1993-01-22',
    'IVV':  '2000-05-15',
    'SPYM': '2005-11-08',
    'UPRO': '2009-06-25',
    'VOO':  '2010-09-07',
    'SSO':  '2006-06-19'
}};

const IPO_NAMES = {{
    'SPY':  'SPY',
    'IVV':  'IVV',
    'SPYM': 'SPYM (原SPLG)',
    'UPRO': 'UPRO (3倍)',
    'VOO':  'VOO',
    'SSO':  'SSO (2倍)'
}};

const PERIODS = [
    {{ label: '1993-2000', start: 1993, end: 2000 }},
    {{ label: '2001-2010', start: 2001, end: 2010 }},
    {{ label: '2011-2020', start: 2011, end: 2020 }},
    {{ label: '2021-至今',  start: 2021, end: new Date().getFullYear() }}
];

function initUI() {{
    const daySel = document.getElementById('startDay');
    for (let d = 1; d <= 31; d++) {{
        const opt = document.createElement('option');
        opt.value = d; opt.text = d + '日';
        daySel.appendChild(opt);
    }}
    updateYearOptions(2021);
    if (RAW_DATA.length > 0) {{
        document.getElementById('endDate').value = RAW_DATA[RAW_DATA.length - 1].date;
    }}
    updateIpoHint();
    onStrategyChange();
    runStrategy();
}}

/* ★ 策略切换时更新UI标签 */
function onStrategyChange() {{
    const mode = document.getElementById('strategyMode').value;
    if (mode === 'dca') {{
        document.getElementById('chkBaseText').innerText = '基础定投';
        document.getElementById('chkBearText').innerText = '熊市加倍';
        document.getElementById('chkExtremeText').innerText = '极端行情';
        document.getElementById('lblChkBase').title = '每月固定日期定投基础金额';
        document.getElementById('lblChkBear').title = '当价格低于MA200时，当月定投金额翻倍';
        document.getElementById('lblChkExtreme').title = '牛市RSI<30切UPRO → 分段减仓 → 跌破MA200清仓';
        document.getElementById('lblCount1Title').innerText = '基础定投期数';
        document.getElementById('lblCount1Sub').innerText = '牛市单倍';
        document.getElementById('lblCount2Title').innerText = '熊市倍投期数';
        document.getElementById('lblCount2Sub').innerText = '熊市双倍';
    }} else {{
        document.getElementById('chkBaseText').innerText = '基础抄底';
        document.getElementById('chkBearText').innerText = '月份追投';
        document.getElementById('chkExtremeText').innerText = '标的切换';
        document.getElementById('lblChkBase').title = 'RSI(10)<30时按天投入基准金额';
        document.getElementById('lblChkBear').title = '按月份进度补齐当年抄底次数';
        document.getElementById('lblChkExtreme').title = '与极端行情相同的4节点减仓+破MA200清仓';
        document.getElementById('lblCount1Title').innerText = '抄底投资天数';
        document.getElementById('lblCount1Sub').innerText = 'RSI<30触发';
        document.getElementById('lblCount2Title').innerText = '月份追投次数';
        document.getElementById('lblCount2Sub').innerText = '补齐月度进度';
    }}
    runStrategy();
}}

function onPeriodChange() {{
    updateYearOptions(null);
    onDatePartChange();
}}

function updateYearOptions(defaultYear) {{
    const pIdx = parseInt(document.getElementById('periodSelect').value);
    const p = PERIODS[pIdx];
    const ySel = document.getElementById('startYear');
    ySel.innerHTML = '';
    for (let y = p.start; y <= p.end; y++) {{
        const opt = document.createElement('option');
        opt.value = y; opt.text = y;
        ySel.appendChild(opt);
    }}
    if (defaultYear && defaultYear >= p.start && defaultYear <= p.end) {{
        ySel.value = defaultYear;
    }}
}}

function onDatePartChange() {{
    validateTickerVsDate(false, 'tickerSelect');
    validateTickerVsDate(false, 'tickerSelect2');
    runStrategy();
}}

function onTickerChange(which) {{
    const selId = which === 1 ? 'tickerSelect' : 'tickerSelect2';
    validateTickerVsDate(true, selId);
    updateIpoHint();
    runStrategy();
}}

function updateIpoHint() {{
    const t1 = document.getElementById('tickerSelect').value;
    document.getElementById('ipoHint').innerHTML =
        IPO_NAMES[t1] + ' 上市: ' + IPO_DATES[t1] + ' | UPRO: ' + IPO_DATES['UPRO'];
    const t2 = document.getElementById('tickerSelect2').value;
    if (t2 !== 'NONE') {{
        document.getElementById('ipoHint2').innerHTML = IPO_NAMES[t2] + ' 上市: ' + IPO_DATES[t2];
    }} else {{
        document.getElementById('ipoHint2').innerHTML = '';
    }}
}}

function validateTickerVsDate(fromTicker, selId) {{
    const sel = document.getElementById(selId);
    const ticker = sel.value;
    if (ticker === 'NONE') return;
    const year   = parseInt(document.getElementById('startYear').value);
    const month  = parseInt(document.getElementById('startMonth').value);
    const day    = parseInt(document.getElementById('startDay').value);
    const dim = new Date(year, month, 0).getDate();
    const d = Math.min(day, dim);
    const startStr = year + '-' + String(month).padStart(2,'0') + '-' + String(d).padStart(2,'0');
    const ipoDate = IPO_DATES[ticker];
    if (startStr < ipoDate) {{
        if (fromTicker) {{
            if (selId === 'tickerSelect') {{
                alert(IPO_NAMES[ticker] + ' 上市时间为 ' + ipoDate + '，晚于当前开始日期。\\n已自动切换为 SPY。');
                sel.value = 'SPY';
            }} else {{
                alert(IPO_NAMES[ticker] + ' 上市时间为 ' + ipoDate + '，晚于当前开始日期。\\n已取消对比。');
                sel.value = 'NONE';
            }}
            updateIpoHint();
        }} else {{
            if (selId === 'tickerSelect' && ticker !== 'SPY') {{
                alert('当前开始日期早于 ' + IPO_NAMES[ticker] + ' 上市时间。\\n标的①已切换为 SPY。');
                sel.value = 'SPY';
                updateIpoHint();
            }}
            if (selId === 'tickerSelect2' && ticker !== 'NONE') {{
                sel.value = 'NONE';
                updateIpoHint();
            }}
        }}
    }}
}}

function buildDcaDateStr(y, m, preferredDay) {{
    const dim = new Date(y, m, 0).getDate();
    const d = Math.min(preferredDay, dim);
    return y + '-' + String(m).padStart(2,'0') + '-' + String(d).padStart(2,'0');
}}

function advanceMonth(y, m) {{
    m++;
    if (m > 12) {{ m = 1; y++; }}
    return [y, m];
}}

function solveMonthlyRate(N, PMT, FV) {{
    if (N <= 1 || PMT <= 0 || FV <= 0) return 0;
    if (Math.abs(FV - PMT * N) < 0.01) return 0;
    let r = (FV / (PMT * N) - 1) * 2 / N;
    if (r === 0) r = 0.001;
    if (r < -0.5) r = -0.01;
    for (let iter = 0; iter < 200; iter++) {{
        const r1 = 1 + r;
        if (r1 <= 0) {{ r = r / 2 + 0.001; continue; }}
        const pow_N   = Math.pow(r1, N);
        const pow_N_1 = Math.pow(r1, N - 1);
        const f  = PMT * (pow_N - 1) / r - FV;
        const df = PMT * (N * r * pow_N_1 - pow_N + 1) / (r * r);
        if (Math.abs(df) < 1e-15) break;
        const rNew = r - f / df;
        if (Math.abs(rNew - r) < 1e-12) {{ r = rNew; break; }}
        r = rNew;
        if (r < -0.99) r = -0.5;
        if (r > 10) r = 1;
    }}
    return r;
}}

function getPositionLabel(position, tickerSymbol) {{
    switch(position) {{
        case 'MAIN':     return {{ text: '🟢 ' + tickerSymbol, cls: 'pos-main' }};
        case 'UPRO100':  return {{ text: '🔴 100% UPRO', cls: 'pos-upro100' }};
        case 'UPRO75':   return {{ text: '🟠 75% UPRO', cls: 'pos-upro75' }};
        case 'UPRO50':   return {{ text: '🟡 50% UPRO', cls: 'pos-upro50' }};
        case 'UPRO25':   return {{ text: '🟤 25% UPRO', cls: 'pos-upro25' }};
        case 'SPY_REFUGE': return {{ text: '🔵 避险SPY', cls: 'pos-main' }};
        default:         return {{ text: position, cls: '' }};
    }}
}}

let valChart = null;
let techChart = null;

/* ════════════════════════════════════════════════
 *  核心: 对单个标的执行策略, 返回结果对象
 *  strategyMode: 'dca' | 'bottom'
 * ════════════════════════════════════════════════ */
function executeStrategy(tickerSymbol, startDateStr, viewEndDate, baseInvest, preferredDay,
                          useBase, useBear, useExtreme, startYear, startMonth, strategyMode) {{

    const UPRO_IPO = IPO_DATES['UPRO'];
    const isMainUpro = (tickerSymbol === 'UPRO');

    // ★ 减仓后的避险标的: 如果主标的是UPRO，减仓后转换为SPY
    const refugeTicker = isMainUpro ? 'SPY' : tickerSymbol;

    let nextDcaYear  = startYear;
    let nextDcaMonth = startMonth;
    let nextDcaDateStr = buildDcaDateStr(nextDcaYear, nextDcaMonth, preferredDay);

    let state = {{
        total_invested: 0,
        main_shares: 0,       // 主标的份额
        upro_shares: 0,       // UPRO(杠杆)份额
        spy_refuge_shares: 0, // ★ SPY避险份额 (仅当主标的=UPRO时使用)
        position: 'MAIN',
        upro_entry_shares: 0, // 用于非UPRO标的的极端行情入场记录
        main_entry_shares: 0, // ★ 用于UPRO标的减仓入场记录
        base_count: 0, bear_count: 0,
        sold_70: false, sold_80: false, sold_85: false, sold_60_drop: false, sell_count: 0
    }};

    // ★ 抄底策略专用状态
    let bottomState = {{
        yearlyInvestCount: 0,
        currentYear: startYear,
        prevDayRsiBelow30: false
    }};

    let logs = [];
    let chartDates=[], chartVals=[];
    let chartIdx=[], chartMAs=[], chartRSIs=[];
    let chartRsiBuy=[], chartRsiWarn=[], chartRsiSell=[], chartRsi85=[], chartRsiDrop60=[];
    let prevRsi = null;

    for (let i = 0; i < RAW_DATA.length; i++) {{
        const row = RAW_DATA[i];
        const rsi = row.RSI_10;
        if (row.date < startDateStr) {{
            if (rsi != null) prevRsi = rsi;
            continue;
        }}
        if (row.date > viewEndDate) break;

        const idxPrice  = row.Index_Close;
        const ma200     = row.Index_MA200;
        const uproPrice = row.UPRO;
        const mainPrice = row[tickerSymbol];
        const spyPrice  = row['SPY']; // ★ 始终需要SPY价格用于避险

        if (!idxPrice || !ma200 || rsi == null || !mainPrice) {{
            chartDates.push(row.date);
            chartVals.push(null); chartIdx.push(null); chartMAs.push(null); chartRSIs.push(null);
            chartRsiBuy.push(null); chartRsiWarn.push(null); chartRsiSell.push(null); chartRsi85.push(null); chartRsiDrop60.push(null);
            if (rsi != null) prevRsi = rsi;
            continue;
        }}

        const isBull = idxPrice > ma200;
        let action = "", note = "", investAmt = 0;

        // ★ 解析当前日期的月份和年份
        const dateYear = parseInt(row.date.substring(0, 4));
        const dateMonth = parseInt(row.date.substring(5, 7));

        /* ═══════════════════════════════════════
         *  定投策略 (DCA) 模式
         * ═══════════════════════════════════════ */
        if (strategyMode === 'dca') {{
            /* ── 基础定投 ── */
            if (useBase) {{
                let dcaThisDay = 0, dcaNotes = [];
                while (row.date >= nextDcaDateStr) {{
                    let currentInvest = baseInvest;
                    if (useBear && !isBull) {{
                        currentInvest = baseInvest * 2;
                        dcaNotes.push("熊市双倍"); state.bear_count++;
                    }} else {{
                        dcaNotes.push("基础定投"); state.base_count++;
                    }}
                    state.total_invested += currentInvest;
                    dcaThisDay += currentInvest;
                    state.main_shares += (currentInvest / USD_HKD) / mainPrice;
                    [nextDcaYear, nextDcaMonth] = advanceMonth(nextDcaYear, nextDcaMonth);
                    nextDcaDateStr = buildDcaDateStr(nextDcaYear, nextDcaMonth, preferredDay);
                }}
                if (dcaThisDay > 0) {{
                    action = "定投"; investAmt = dcaThisDay; note = dcaNotes.join(" + ");
                }}
            }}

            /* ── 极端行情 (统一处理 UPRO和非UPRO) ── */
            const uproLive = (row.date >= UPRO_IPO) && uproPrice;
            if (useExtreme && uproLive) {{
                if (!isMainUpro) {{
                    /* --- 非UPRO标的: 牛市RSI<30 → 全仓切UPRO --- */
                    if (state.main_shares > 0 && isBull && rsi < 30) {{
                        const valMain = state.main_shares * mainPrice;
                        const newUproShares = valMain / uproPrice;
                        state.upro_shares += newUproShares;
                        state.upro_entry_shares = state.upro_shares;
                        state.main_shares = 0;
                        state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        state.position = 'UPRO100';
                        action = action ? action + " + 全仓切UPRO" : "全仓切UPRO";
                        note += (note?" | ":"") + "牛市RSI="+rsi.toFixed(1)+"<30 抄底";
                    }}
                    else if (state.position !== 'MAIN' && state.upro_shares > 0) {{
                        const entryShares = state.upro_entry_shares;
                        let chunksToSell = 0, notesArr = [];
                        if (!state.sold_70 && rsi > 70) {{ state.sold_70=true; chunksToSell++; notesArr.push("RSI>70"); }}
                        if (!state.sold_80 && rsi > 80) {{ state.sold_80=true; chunksToSell++; notesArr.push("RSI>80"); }}
                        if (!state.sold_85 && rsi > 85) {{ state.sold_85=true; chunksToSell++; notesArr.push("RSI>85"); }}
                        if (state.sold_70 && !state.sold_60_drop && prevRsi!=null && prevRsi>=60 && rsi<60) {{
                            state.sold_60_drop=true; chunksToSell++; notesArr.push("RSI回落破60");
                        }}
                        if (chunksToSell > 0) {{
                            const sellShares = Math.min(entryShares*0.25*chunksToSell, state.upro_shares);
                            if (sellShares > 0) {{
                                state.upro_shares -= sellShares;
                                state.main_shares += (sellShares*uproPrice)/mainPrice;
                                state.sell_count += chunksToSell;
                                action = action ? action+" + 减仓"+(chunksToSell*25)+"%" : "减仓"+(chunksToSell*25)+"%";
                                note += (note?" | ":"") + notesArr.join(", ");
                                if (state.sell_count===1) state.position='UPRO75';
                                else if (state.sell_count===2) state.position='UPRO50';
                                else if (state.sell_count===3) state.position='UPRO25';
                                else state.position='MAIN';
                            }}
                        }}
                        if (state.position!=='MAIN' && state.upro_shares>0 && state.sold_70 && !isBull) {{
                            state.main_shares += (state.upro_shares*uproPrice)/mainPrice;
                            state.upro_shares=0; state.position='MAIN';
                            action = action ? action+" + 破MA200清仓" : "破MA200清仓";
                            note += (note?" | ":"") + "跌破MA200 剩余→"+tickerSymbol;
                        }}
                        if (state.upro_shares<=0 && state.position!=='MAIN') {{
                            state.upro_shares=0; state.position='MAIN'; state.upro_entry_shares=0;
                            state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        }}
                    }}
                }} else {{
                    /* --- ★ UPRO标的: RSI>70等节点减仓→SPY, 牛市RSI<30从SPY转回UPRO --- */
                    if (state.spy_refuge_shares > 0 && isBull && rsi < 30 && spyPrice) {{
                        // 从SPY避险回归UPRO
                        const valSpy = state.spy_refuge_shares * spyPrice;
                        state.main_shares += valSpy / mainPrice; // mainPrice就是UPRO价格
                        state.spy_refuge_shares = 0;
                        state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        state.position = 'MAIN';
                        action = action ? action + " + SPY转回UPRO" : "SPY转回UPRO";
                        note += (note?" | ":"") + "牛市RSI="+rsi.toFixed(1)+"<30 回归杠杆";
                    }}
                    else if (state.main_shares > 0 && spyPrice) {{
                        // UPRO减仓 → SPY
                        if (state.sell_count === 0 && state.main_shares > 0) {{
                            state.main_entry_shares = state.main_shares; // 记录入场份额
                        }}
                        const entryShares = state.main_entry_shares > 0 ? state.main_entry_shares : state.main_shares;
                        let chunksToSell = 0, notesArr = [];
                        if (!state.sold_70 && rsi > 70) {{ state.sold_70=true; chunksToSell++; notesArr.push("RSI>70"); }}
                        if (!state.sold_80 && rsi > 80) {{ state.sold_80=true; chunksToSell++; notesArr.push("RSI>80"); }}
                        if (!state.sold_85 && rsi > 85) {{ state.sold_85=true; chunksToSell++; notesArr.push("RSI>85"); }}
                        if (state.sold_70 && !state.sold_60_drop && prevRsi!=null && prevRsi>=60 && rsi<60) {{
                            state.sold_60_drop=true; chunksToSell++; notesArr.push("RSI回落破60");
                        }}
                        if (chunksToSell > 0) {{
                            const sellShares = Math.min(entryShares*0.25*chunksToSell, state.main_shares);
                            if (sellShares > 0) {{
                                state.main_shares -= sellShares;
                                state.spy_refuge_shares += (sellShares * mainPrice) / spyPrice;
                                state.sell_count += chunksToSell;
                                action = action ? action+" + UPRO减仓"+(chunksToSell*25)+"%→SPY" : "UPRO减仓"+(chunksToSell*25)+"%→SPY";
                                note += (note?" | ":"") + notesArr.join(", ");
                                if (state.main_shares > 0) {{
                                    if (state.sell_count===1) state.position='UPRO75';
                                    else if (state.sell_count===2) state.position='UPRO50';
                                    else if (state.sell_count===3) state.position='UPRO25';
                                    else state.position='SPY_REFUGE';
                                }} else {{
                                    state.position='SPY_REFUGE';
                                }}
                            }}
                        }}
                        // 破MA200清仓: 剩余UPRO全部转SPY
                        if (state.position!=='MAIN' && state.position!=='SPY_REFUGE' && state.main_shares>0 && state.sold_70 && !isBull && spyPrice) {{
                            state.spy_refuge_shares += (state.main_shares * mainPrice) / spyPrice;
                            state.main_shares=0; state.position='SPY_REFUGE';
                            action = action ? action+" + 破MA200清仓→SPY" : "破MA200清仓→SPY";
                            note += (note?" | ":"") + "跌破MA200 UPRO→SPY";
                        }}
                        if (state.main_shares<=0 && state.spy_refuge_shares > 0 && state.position!=='SPY_REFUGE') {{
                            state.position='SPY_REFUGE';
                        }}
                        if (state.main_shares<=0 && state.spy_refuge_shares<=0) {{
                            state.position='MAIN'; state.main_entry_shares=0;
                            state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        }}
                    }}
                }}
            }}
        }}

        /* ═══════════════════════════════════════
         *  抄底策略 (BOTTOM) 模式
         * ═══════════════════════════════════════ */
        else if (strategyMode === 'bottom') {{
            // ★ 年度重置
            if (dateYear !== bottomState.currentYear) {{
                bottomState.yearlyInvestCount = 0;
                bottomState.currentYear = dateYear;
            }}

            /* ── 基础抄底: RSI<30时按天投入 ── */
            if (useBase && rsi < 30) {{
                let investMultiple = 1;
                let isNewEpisode = !bottomState.prevDayRsiBelow30;

                // ★ 月份追投: 新一轮RSI<30首日, 补齐月度进度
                if (useBear && isNewEpisode) {{
                    let deficit = dateMonth - bottomState.yearlyInvestCount;
                    if (deficit > 1) {{
                        investMultiple = deficit;
                        state.bear_count += (deficit - 1); // 追投次数
                    }}
                }}

                let investAmount = investMultiple * baseInvest;
                state.total_invested += investAmount;
                state.main_shares += (investAmount / USD_HKD) / mainPrice;
                bottomState.yearlyInvestCount += investMultiple;
                state.base_count++; // 抄底天数

                investAmt = investAmount;
                if (investMultiple > 1) {{
                    action = "抄底+追投" + investMultiple + "x";
                    note = "RSI=" + rsi.toFixed(1) + "<30 | 月份:" + dateMonth + " 年累计:" + bottomState.yearlyInvestCount;
                }} else {{
                    action = "抄底";
                    note = "RSI=" + rsi.toFixed(1) + "<30 | 年累计:" + bottomState.yearlyInvestCount;
                }}
            }}

            bottomState.prevDayRsiBelow30 = (rsi < 30);

            /* ── 标的切换 (与极端行情相同的减仓逻辑) ── */
            const uproLive = (row.date >= UPRO_IPO) && uproPrice;
            if (useExtreme && uproLive) {{
                if (!isMainUpro) {{
                    /* --- 非UPRO标的: 牛市RSI<30 → 全仓切UPRO --- */
                    if (state.main_shares > 0 && isBull && rsi < 30) {{
                        const valMain = state.main_shares * mainPrice;
                        const newUproShares = valMain / uproPrice;
                        state.upro_shares += newUproShares;
                        state.upro_entry_shares = state.upro_shares;
                        state.main_shares = 0;
                        state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        state.position = 'UPRO100';
                        action = action ? action + " + 全仓切UPRO" : "全仓切UPRO";
                        note += (note?" | ":"") + "牛市RSI<30 切杠杆";
                    }}
                    else if (state.position !== 'MAIN' && state.upro_shares > 0) {{
                        const entryShares = state.upro_entry_shares;
                        let chunksToSell = 0, notesArr = [];
                        if (!state.sold_70 && rsi > 70) {{ state.sold_70=true; chunksToSell++; notesArr.push("RSI>70"); }}
                        if (!state.sold_80 && rsi > 80) {{ state.sold_80=true; chunksToSell++; notesArr.push("RSI>80"); }}
                        if (!state.sold_85 && rsi > 85) {{ state.sold_85=true; chunksToSell++; notesArr.push("RSI>85"); }}
                        if (state.sold_70 && !state.sold_60_drop && prevRsi!=null && prevRsi>=60 && rsi<60) {{
                            state.sold_60_drop=true; chunksToSell++; notesArr.push("RSI回落破60");
                        }}
                        if (chunksToSell > 0) {{
                            const sellShares = Math.min(entryShares*0.25*chunksToSell, state.upro_shares);
                            if (sellShares > 0) {{
                                state.upro_shares -= sellShares;
                                state.main_shares += (sellShares*uproPrice)/mainPrice;
                                state.sell_count += chunksToSell;
                                action = action ? action+" + 减仓"+(chunksToSell*25)+"%" : "减仓"+(chunksToSell*25)+"%";
                                note += (note?" | ":"") + notesArr.join(", ");
                                if (state.sell_count===1) state.position='UPRO75';
                                else if (state.sell_count===2) state.position='UPRO50';
                                else if (state.sell_count===3) state.position='UPRO25';
                                else state.position='MAIN';
                            }}
                        }}
                        if (state.position!=='MAIN' && state.upro_shares>0 && state.sold_70 && !isBull) {{
                            state.main_shares += (state.upro_shares*uproPrice)/mainPrice;
                            state.upro_shares=0; state.position='MAIN';
                            action = action ? action+" + 破MA200清仓" : "破MA200清仓";
                            note += (note?" | ":"") + "跌破MA200 剩余→"+tickerSymbol;
                        }}
                        if (state.upro_shares<=0 && state.position!=='MAIN') {{
                            state.upro_shares=0; state.position='MAIN'; state.upro_entry_shares=0;
                            state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        }}
                    }}
                }} else {{
                    /* --- ★ UPRO标的: 减仓→SPY, 牛市RSI<30从SPY转回UPRO --- */
                    if (state.spy_refuge_shares > 0 && isBull && rsi < 30 && spyPrice) {{
                        const valSpy = state.spy_refuge_shares * spyPrice;
                        state.main_shares += valSpy / mainPrice;
                        state.spy_refuge_shares = 0;
                        state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        state.position = 'MAIN';
                        action = action ? action + " + SPY转回UPRO" : "SPY转回UPRO";
                        note += (note?" | ":"") + "牛市RSI<30 回归杠杆";
                    }}
                    else if (state.main_shares > 0 && spyPrice) {{
                        if (state.sell_count === 0 && state.main_shares > 0) {{
                            state.main_entry_shares = state.main_shares;
                        }}
                        const entryShares = state.main_entry_shares > 0 ? state.main_entry_shares : state.main_shares;
                        let chunksToSell = 0, notesArr = [];
                        if (!state.sold_70 && rsi > 70) {{ state.sold_70=true; chunksToSell++; notesArr.push("RSI>70"); }}
                        if (!state.sold_80 && rsi > 80) {{ state.sold_80=true; chunksToSell++; notesArr.push("RSI>80"); }}
                        if (!state.sold_85 && rsi > 85) {{ state.sold_85=true; chunksToSell++; notesArr.push("RSI>85"); }}
                        if (state.sold_70 && !state.sold_60_drop && prevRsi!=null && prevRsi>=60 && rsi<60) {{
                            state.sold_60_drop=true; chunksToSell++; notesArr.push("RSI回落破60");
                        }}
                        if (chunksToSell > 0) {{
                            const sellShares = Math.min(entryShares*0.25*chunksToSell, state.main_shares);
                            if (sellShares > 0) {{
                                state.main_shares -= sellShares;
                                state.spy_refuge_shares += (sellShares * mainPrice) / spyPrice;
                                state.sell_count += chunksToSell;
                                action = action ? action+" + UPRO减仓"+(chunksToSell*25)+"%→SPY" : "UPRO减仓"+(chunksToSell*25)+"%→SPY";
                                note += (note?" | ":"") + notesArr.join(", ");
                                if (state.main_shares > 0) {{
                                    if (state.sell_count===1) state.position='UPRO75';
                                    else if (state.sell_count===2) state.position='UPRO50';
                                    else if (state.sell_count===3) state.position='UPRO25';
                                    else state.position='SPY_REFUGE';
                                }} else {{
                                    state.position='SPY_REFUGE';
                                }}
                            }}
                        }}
                        if (state.position!=='MAIN' && state.position!=='SPY_REFUGE' && state.main_shares>0 && state.sold_70 && !isBull && spyPrice) {{
                            state.spy_refuge_shares += (state.main_shares * mainPrice) / spyPrice;
                            state.main_shares=0; state.position='SPY_REFUGE';
                            action = action ? action+" + 破MA200清仓→SPY" : "破MA200清仓→SPY";
                            note += (note?" | ":"") + "跌破MA200 UPRO→SPY";
                        }}
                        if (state.main_shares<=0 && state.spy_refuge_shares > 0 && state.position!=='SPY_REFUGE') {{
                            state.position='SPY_REFUGE';
                        }}
                        if (state.main_shares<=0 && state.spy_refuge_shares<=0) {{
                            state.position='MAIN'; state.main_entry_shares=0;
                            state.sold_70=false; state.sold_80=false; state.sold_85=false; state.sold_60_drop=false; state.sell_count=0;
                        }}
                    }}
                }}
            }}
        }}

        /* ═══ 计算总市值 ═══ */
        let valAssets = state.main_shares * mainPrice;
        valAssets += state.upro_shares * (uproPrice || 0);
        if (state.spy_refuge_shares > 0 && spyPrice) {{
            valAssets += state.spy_refuge_shares * spyPrice;
        }}
        const totalValHkd = valAssets * USD_HKD;

        const isLastRow = (i === RAW_DATA.length - 1) || (row.date === viewEndDate);
        if (action !== "" || isLastRow) {{
            logs.push({{
                date: row.date, idx: idxPrice, rsi: rsi, isBull: isBull,
                invest: investAmt, totalInv: state.total_invested,
                totalVal: totalValHkd, mainS: state.main_shares, uproS: state.upro_shares,
                spyRefS: state.spy_refuge_shares,
                action: action, note: note, position: state.position,
                baseCount: state.base_count, bearCount: state.bear_count
            }});
        }}

        chartDates.push(row.date);
        chartVals.push(totalValHkd);
        chartIdx.push(idxPrice); chartMAs.push(ma200); chartRSIs.push(rsi);

        let valBuy=null, valWarn=null, valSell=null, val85=null, valDrop60=null;
        if (prevRsi !== null) {{
            if (rsi<30 && prevRsi>=30 && isBull) valBuy=rsi;
            if (rsi>70 && prevRsi<=70) valWarn=rsi;
            if (rsi>80 && prevRsi<=80) valSell=rsi;
            if (rsi>85 && prevRsi<=85) val85=rsi;
            if (rsi<60 && prevRsi>=60) valDrop60=rsi;
        }}
        chartRsiBuy.push(valBuy); chartRsiWarn.push(valWarn); chartRsiSell.push(valSell);
        chartRsi85.push(val85); chartRsiDrop60.push(valDrop60);
        prevRsi = rsi;
    }}

    return {{
        logs, chartDates, chartVals, chartIdx, chartMAs, chartRSIs,
        chartRsiBuy, chartRsiWarn, chartRsiSell, chartRsi85, chartRsiDrop60
    }};
}}


/* ════════════════════════════════════════════════
 *  主入口: 运行策略 (支持PK模式)
 * ════════════════════════════════════════════════ */
function runStrategy() {{
    const startYear   = parseInt(document.getElementById('startYear').value);
    const startMonth  = parseInt(document.getElementById('startMonth').value);
    const startDayReq = parseInt(document.getElementById('startDay').value);
    const baseInvest  = parseFloat(document.getElementById('baseInvest').value);
    let viewEndDate   = document.getElementById('endDate').value;
    const ticker1     = document.getElementById('tickerSelect').value;
    const ticker2     = document.getElementById('tickerSelect2').value;
    const strategyMode = document.getElementById('strategyMode').value;

    const useBase    = document.getElementById('chkBase').checked;
    const useBear    = document.getElementById('chkBear').checked;
    const useExtreme = document.getElementById('chkExtreme').checked;

    document.getElementById('lblTicker').innerText = ticker1;
    document.getElementById('lblTicker1Tag').innerText = '① ' + ticker1;
    document.getElementById('lblShareTicker').innerText = ticker1;
    document.getElementById('thShareTicker').innerText = ticker1;

    const daysInMonth = new Date(startYear, startMonth, 0).getDate();
    const actualDay = Math.min(startDayReq, daysInMonth);
    const startDateStr = startYear + '-' + String(startMonth).padStart(2,'0') + '-' + String(actualDay).padStart(2,'0');
    if (!viewEndDate && RAW_DATA.length > 0) viewEndDate = RAW_DATA[RAW_DATA.length - 1].date;

    const r1 = executeStrategy(ticker1, startDateStr, viewEndDate, baseInvest, startDayReq,
                                useBase, useBear, useExtreme, startYear, startMonth, strategyMode);

    let r2 = null;
    const hasPK = ticker2 !== 'NONE';
    if (hasPK) {{
        r2 = executeStrategy(ticker2, startDateStr, viewEndDate, baseInvest, startDayReq,
                              useBase, useBear, useExtreme, startYear, startMonth, strategyMode);
    }}

    updateStatusCards(r1.logs, RAW_DATA, baseInvest, ticker1, useBase, useBear, strategyMode);
    updateTable(r1.logs, ticker1);
    updateCharts(r1, r2, ticker1, ticker2);
    updatePKBox(r1, r2, ticker1, ticker2, hasPK);
}}


function updateStatusCards(logs, rawData, baseInvest, tickerSymbol, useBase, useBear, strategyMode) {{
    const lastLog = logs[logs.length - 1];
    if (!lastLog) return;

    const rawRow = rawData.find(r => r.date === lastLog.date);
    const ma200 = rawRow ? rawRow.Index_MA200 : 0;
    const isBull = lastLog.idx > ma200;

    document.getElementById('valIdx').innerText = lastLog.idx.toFixed(2);
    document.getElementById('valDate').innerText = lastLog.date;
    document.getElementById('valMa200').innerText = ma200.toFixed(2);
    const rsiEl = document.getElementById('valRsi');
    rsiEl.innerText = lastLog.rsi.toFixed(1);
    rsiEl.style.color = lastLog.rsi < 30 ? 'var(--bear)' : (lastLog.rsi > 70 ? 'var(--warn)' : 'var(--bull)');

    let rsiSig = "正常";
    if (lastLog.rsi < 30) rsiSig = isBull ? "⚠️ 牛市急跌(准备)" : "📉 熊市超卖";
    if (lastLog.rsi > 70) rsiSig = "🔵 超买警戒";
    if (lastLog.rsi > 80) rsiSig = "🟢 极度超买";
    if (lastLog.rsi > 85) rsiSig = "🔥 超极端";
    document.getElementById('valRsiSig').innerText = rsiSig;

    document.getElementById('valTotalInv').innerText = Math.round(lastLog.totalInv).toLocaleString();
    document.getElementById('valTotalVal').innerText = Math.round(lastLog.totalVal).toLocaleString();
    document.getElementById('valBaseCount').innerText = lastLog.baseCount;
    document.getElementById('valBearCount').innerText = lastLog.bearCount;

    const returnRate = lastLog.totalInv > 0 ? ((lastLog.totalVal - lastLog.totalInv) / lastLog.totalInv * 100) : 0;
    document.getElementById('statusReturn').innerText = (returnRate > 0 ? "+" : "") + returnRate.toFixed(2) + "%";
    document.getElementById('statusReturn').className = "action-val " + (returnRate >= 0 ? "c-bull" : "c-bear");

    // ★ 年化收益率: 定投用月度N, 抄底用投资天数N
    const N = lastLog.baseCount + lastLog.bearCount;
    const PMT = N > 0 ? lastLog.totalInv / N : 0;
    const FV = lastLog.totalVal;
    let annualRate = 0;
    if (strategyMode === 'dca') {{
        // 月度定投年化
        if (N > 1 && PMT > 0 && FV > 0) {{
            annualRate = solveMonthlyRate(N, PMT, FV) * 12 * 100;
        }}
    }} else {{
        // 抄底策略: 用总投入天数近似
        if (N > 1 && PMT > 0 && FV > 0) {{
            annualRate = solveMonthlyRate(N, PMT, FV) * 12 * 100;
        }}
    }}
    const annEl = document.getElementById('statusAnnual');
    annEl.innerText = (annualRate > 0 ? "+" : "") + annualRate.toFixed(2) + "%";
    annEl.className = "action-val " + (annualRate >= 0 ? "c-bull" : "c-bear");
    document.getElementById('annN').innerText = N;
    document.getElementById('annPMT').innerText = Math.round(PMT).toLocaleString();

    document.getElementById('shareSpy').innerText = lastLog.mainS.toFixed(2);
    document.getElementById('shareUpro').innerText = lastLog.uproS.toFixed(2);

    // ★ 显示SPY避险份额 (UPRO标的时)
    const isMainUpro = (tickerSymbol === 'UPRO');
    const spyRefEl = document.getElementById('shareSpyRefuge');
    if (isMainUpro && lastLog.spyRefS > 0) {{
        spyRefEl.style.display = 'block';
        document.getElementById('shareSpyRef').innerText = lastLog.spyRefS.toFixed(2);
    }} else {{
        spyRefEl.style.display = 'none';
    }}

    const posInfo = getPositionLabel(lastLog.position, tickerSymbol);
    document.getElementById('statusPos').innerHTML = '<span class="pos-tag ' + posInfo.cls + '">' + posInfo.text + '</span>';

    let posDetail = "";
    if (isMainUpro && lastLog.spyRefS > 0) {{
        posDetail = "UPRO " + lastLog.mainS.toFixed(2) + "股 + SPY(避险) " + lastLog.spyRefS.toFixed(2) + "股";
    }} else if (lastLog.position !== 'MAIN' && lastLog.uproS > 0) {{
        posDetail = "UPRO " + lastLog.uproS.toFixed(2) + "股 + " + tickerSymbol + " " + lastLog.mainS.toFixed(2) + "股";
    }} else {{
        posDetail = tickerSymbol + " " + lastLog.mainS.toFixed(2) + "股";
    }}
    document.getElementById('statusPosDetail').innerText = posDetail;

    // ★ 下期建议
    if (strategyMode === 'dca') {{
        let nextInv = 0;
        if (useBase) {{ nextInv = (useBear && !isBull) ? baseInvest * 2 : baseInvest; }}
        document.getElementById('statusNextInvest').innerText = "HK$ " + nextInv.toLocaleString();
    }} else {{
        const lastRsi = lastLog.rsi;
        if (lastRsi < 30) {{
            document.getElementById('statusNextInvest').innerText = "HK$ " + baseInvest.toLocaleString() + "/天";
        }} else {{
            document.getElementById('statusNextInvest').innerText = "等待 RSI<30";
        }}
    }}

    const marketText = isBull ? "🐂 牛市" : "🐻 熊市";
    document.getElementById('statusMarket').innerHTML = '<span class="' + (isBull?'c-bull':'c-bear') + '">' + marketText + '</span>';
}}


function updatePKBox(r1, r2, t1, t2, hasPK) {{
    const box = document.getElementById('pkBox');
    if (!hasPK || !r2) {{
        box.classList.remove('active');
        document.getElementById('chartPkHint').innerText = '';
        return;
    }}
    box.classList.add('active');
    document.getElementById('chartPkHint').innerText = '  — ① ' + t1 + ' vs ② ' + t2;

    const last1 = r1.logs[r1.logs.length - 1];
    const last2 = r2.logs[r2.logs.length - 1];
    if (!last1 || !last2) return;

    const ret1 = last1.totalInv > 0 ? ((last1.totalVal - last1.totalInv) / last1.totalInv * 100) : 0;
    const ret2 = last2.totalInv > 0 ? ((last2.totalVal - last2.totalInv) / last2.totalInv * 100) : 0;

    const N1 = last1.baseCount + last1.bearCount;
    const PMT1 = N1 > 0 ? last1.totalInv / N1 : 0;
    const ann1 = (N1 > 1 && PMT1 > 0) ? solveMonthlyRate(N1, PMT1, last1.totalVal) * 12 * 100 : 0;

    const N2 = last2.baseCount + last2.bearCount;
    const PMT2 = N2 > 0 ? last2.totalInv / N2 : 0;
    const ann2 = (N2 > 1 && PMT2 > 0) ? solveMonthlyRate(N2, PMT2, last2.totalVal) * 12 * 100 : 0;

    document.getElementById('pkLabel1').innerText = '① ' + t1;
    document.getElementById('pkLabel2').innerText = '② ' + t2;

    const fmt = (v) => (v>0?'+':'') + v.toFixed(2) + '%';
    document.getElementById('pkReturn1').innerText = fmt(ret1);
    document.getElementById('pkReturn1').className = 'pk-val ' + (ret1>=0?'c-bull':'c-bear');
    document.getElementById('pkReturn2').innerText = fmt(ret2);
    document.getElementById('pkReturn2').className = 'pk-val ' + (ret2>=0?'c-bull':'c-bear');
    document.getElementById('pkAnnual1').innerText = fmt(ann1);
    document.getElementById('pkAnnual2').innerText = fmt(ann2);

    const invDiff = last1.totalInv - last2.totalInv;
    const valDiff = last1.totalVal - last2.totalVal;
    document.getElementById('pkInvDiff').innerText = (invDiff >= 0 ? '+' : '') + Math.round(invDiff).toLocaleString() + ' HKD';
    const vd = document.getElementById('pkValDiff');
    vd.innerText = (valDiff >= 0 ? '+' : '') + Math.round(valDiff).toLocaleString() + ' HKD';
    vd.className = 'pk-val ' + (valDiff >= 0 ? 'c-bull' : 'c-bear');
}}

function updateTable(logs, tickerSymbol) {{
    const tbody = document.querySelector('#logTable tbody');
    tbody.innerHTML = "";
    const displayLogs = logs.slice().reverse().slice(0, 200);
    displayLogs.forEach(l => {{
        const tr = document.createElement('tr');
        if (l.action.includes('UPRO') || l.action.includes('减仓') || l.action.includes('清仓') || l.action.includes('追投')) tr.className = 'highlight';
        const posInfo = getPositionLabel(l.position, tickerSymbol);
        tr.innerHTML =
            '<td>' + l.date + '</td>' +
            '<td>' + l.idx.toFixed(2) + '</td>' +
            '<td class="' + (l.rsi<30?'c-bear':(l.rsi>70?'c-warn':'')) + '">' + l.rsi.toFixed(1) + '</td>' +
            '<td class="' + (l.isBull?'c-bull':'c-bear') + '">' + (l.isBull?'🐂':'🐻') + '</td>' +
            '<td>' + (l.invest > 0 ? '+'+l.invest.toLocaleString() : '') + '</td>' +
            '<td>' + Math.round(l.totalInv).toLocaleString() + '</td>' +
            '<td>' + Math.round(l.totalVal).toLocaleString() + '</td>' +
            '<td>' + l.mainS.toFixed(2) + '</td>' +
            '<td style="color:' + (l.uproS>0?'var(--warn)':'#888') + '">' + l.uproS.toFixed(2) + '</td>' +
            '<td><span class="pos-tag ' + posInfo.cls + '">' + posInfo.text + '</span></td>' +
            '<td><strong>' + l.action + '</strong><div style="font-size:0.8em;color:#aaa">' + l.note + '</div></td>';
        tbody.appendChild(tr);
    }});
}}


function updateCharts(r1, r2, t1, t2) {{
    const dates = r1.chartDates;

    const ctxVal = document.getElementById('mainChart').getContext('2d');
    if (valChart) valChart.destroy();

    const datasets = [{{
        label: '① ' + t1 + ' 市值 (HKD)', data: r1.chartVals,
        borderColor: '#00ff88', backgroundColor: 'rgba(0,255,136,0.08)',
        borderWidth: 2, fill: true, pointRadius: 0, pointHitRadius: 10
    }}];

    if (r2) {{
        const map2 = {{}};
        r2.chartDates.forEach((d,i) => {{ map2[d] = r2.chartVals[i]; }});
        const vals2aligned = dates.map(d => map2[d] !== undefined ? map2[d] : null);
        datasets.push({{
            label: '② ' + t2 + ' 市值 (HKD)', data: vals2aligned,
            borderColor: '#ffaa00', backgroundColor: 'rgba(255,170,0,0.06)',
            borderWidth: 2, fill: true, pointRadius: 0, pointHitRadius: 10,
            borderDash: [6, 3]
        }});
    }}

    valChart = new Chart(ctxVal, {{
        type: 'line',
        data: {{ labels: dates, datasets }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            interaction: {{mode:'index', intersect:false}},
            plugins: {{ legend: {{ display: !!r2, labels: {{ color: '#ccc' }} }} }},
            scales: {{
                x: {{ display: false }},
                y: {{ position:'right', grid:{{color:'rgba(255,255,255,0.05)'}}, ticks:{{color:'#888'}} }}
            }}
        }}
    }});

    const ctxTech = document.getElementById('techChart').getContext('2d');
    if (techChart) techChart.destroy();
    techChart = new Chart(ctxTech, {{
        type: 'line',
        data: {{
            labels: dates,
            datasets: [
                {{ label:'S&P 500 Index', data:r1.chartIdx, borderColor:'#00d4ff', borderWidth:1.5, yAxisID:'y', pointRadius:0 }},
                {{ label:'MA200 (Daily)', data:r1.chartMAs, borderColor:'#ff6b6b', borderWidth:1.5, borderDash:[5,5], yAxisID:'y', pointRadius:0 }},
                {{ label:'RSI(10)', data:r1.chartRSIs, borderColor:'#ffaa00', borderWidth:1, yAxisID:'y1', pointRadius:0 }},
                {{ label:'SigBuy', data:r1.chartRsiBuy, borderColor:'red', backgroundColor:'red', pointStyle:'triangle', rotation:0, pointRadius:6, showLine:false, yAxisID:'y1' }},
                {{ label:'SigWarn70', data:r1.chartRsiWarn, borderColor:'blue', backgroundColor:'blue', pointStyle:'triangle', rotation:180, pointRadius:6, showLine:false, yAxisID:'y1' }},
                {{ label:'SigSell80', data:r1.chartRsiSell, borderColor:'#00ff00', backgroundColor:'#00ff00', pointStyle:'triangle', rotation:180, pointRadius:8, showLine:false, yAxisID:'y1' }},
                {{ label:'SigSell85', data:r1.chartRsi85, borderColor:'#ff00ff', backgroundColor:'#ff00ff', pointStyle:'triangle', rotation:180, pointRadius:8, showLine:false, yAxisID:'y1' }},
                {{ label:'SigDrop60', data:r1.chartRsiDrop60, borderColor:'#ff8800', backgroundColor:'#ff8800', pointStyle:'triangle', rotation:180, pointRadius:7, showLine:false, yAxisID:'y1' }}
            ]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            interaction: {{mode:'index', intersect:false}},
            plugins: {{
                legend: {{
                    labels: {{
                        color:'#ccc',
                        filter: function(item) {{ return !item.text.includes('Sig'); }}
                    }}
                }}
            }},
            scales: {{
                x: {{ ticks:{{color:'#666', maxTicksLimit:10}} }},
                y: {{ type:'linear', display:true, position:'left', grid:{{color:'rgba(255,255,255,0.05)'}}, ticks:{{color:'#888'}} }},
                y1: {{ type:'linear', display:true, position:'right', min:0, max:100, grid:{{display:false}}, ticks:{{color:'#ffaa00'}} }}
            }}
        }}
    }});
}}

initUI();
</script>
</body></html>"""

    return html_content


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("🚀 启动 S&P 500 定投及增益策略")
    print(f"📊 数据源: 指数({INDEX_TICKER}) + 标的{TARGET_ETFS} + 杠杆({LEVERAGE_TICKER})")
    print(f"📊 RSI 固定周期: {RSI_PERIOD}")
    print("="*60)

    try:
        dm = CSVDataManager()
        usd_hkd = get_usd_hkd_rate()
        print(f"💱 当前汇率: {usd_hkd:.4f}")

        print("\n📥 更新数据...")
        index_data = update_ticker_data(dm, INDEX_TICKER, earliest_date=INDEX_START_DATE)
        index_data = calculate_indicators(index_data)

        etf_data_map = {}
        for ticker in TARGET_ETFS:
            if ticker in TICKER_MERGE_MAP:
                etf_data_map[ticker] = update_merged_ticker(dm, ticker, TICKER_MERGE_MAP[ticker])
            else:
                etf_data_map[ticker] = update_ticker_data(dm, ticker)

        upro_data = update_ticker_data(dm, LEVERAGE_TICKER)

        json_data = prepare_json_data(index_data, etf_data_map, upro_data)
        html_str = generate_interactive_html(json_data, usd_hkd)
        with open(HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(html_str)

        print(f"\n✅ 成功! 报表已生成: {HTML_PATH}")
        webbrowser.open('file://' + os.path.realpath(HTML_PATH))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 出错: {e}")

    input("按回车键退出...")