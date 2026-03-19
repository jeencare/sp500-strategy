import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'docs'), exist_ok=True)

SP_INDEX_TICKER  = '^GSPC'
NDX_INDEX_TICKER = '^NDX'
INDEX_START_DATE = '1990-01-01'

SP_ETFS  = ['SPY', 'VOO', 'IVV', 'SPYM', 'SSO', 'UPRO']
NDX_ETFS = ['QQQ', 'QQQM', 'QLD', 'TQQQ']
ALL_ETFS = SP_ETFS + NDX_ETFS

HTML_PATH = os.path.join(BASE_DIR, 'docs', 'index.html')
TICKER_MERGE_MAP = { 'SPYM': ['SPLG', 'SPYM'] }
RSI_PERIOD = 10
IS_CI = os.environ.get('CI', 'false').lower() == 'true'

print(f"📂 数据目录: {DATA_DIR}")
print(f"🤖 CI模式: {IS_CI}")


class CSVDataManager:
    def get_path(self, ticker):
        return os.path.join(DATA_DIR, f"{ticker.replace('^', '')}_daily.csv")

    def load_data(self, ticker):
        path = self.get_path(ticker)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col='date', parse_dates=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
            except:
                return None
        return None

    def save_data(self, df, ticker):
        path = self.get_path(ticker)
        df_save = df.reset_index()
        df_save.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        if 'date' in df_save.columns:
            df_save['date'] = pd.to_datetime(df_save['date']).dt.strftime('%Y-%m-%d')
        df_save.to_csv(path, index=False, encoding='utf-8-sig')


def calculate_indicators(df):
    data = df.copy().sort_index()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    data['RSI_10'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    return data


def get_usd_hkd_rate():
    try:
        fx = yf.Ticker("HKD=X")
        hist = fx.history(period="5d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return 7.8
    except:
        return 7.8


def fetch_new_data(ticker, start_date=None):
    print(f"  ⬇️ 下载 {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date) if start_date else stock.history(period="max")
        if data.empty:
            return pd.DataFrame()
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"  ⚠️ 下载 {ticker} 失败: {e}")
        return pd.DataFrame()


def update_ticker_data(dm, ticker, earliest_date=None):
    old_data = dm.load_data(ticker)
    if old_data is not None and not old_data.empty:
        start_fetch = old_data.index[-1] - timedelta(days=7)
        new_data = fetch_new_data(ticker, start_date=start_fetch)
        full_data = pd.concat([old_data[['Open','High','Low','Close','Volume']], new_data])
        full_data = full_data[~full_data.index.duplicated(keep='last')].sort_index()
    else:
        full_data = fetch_new_data(ticker, start_date=earliest_date)
    dm.save_data(full_data, ticker)
    return full_data


def update_merged_ticker(dm, target_ticker, source_tickers):
    print(f"📦 合并 {source_tickers} → {target_ticker}")
    existing = dm.load_data(target_ticker)
    all_frames = []
    if existing is not None and not existing.empty:
        all_frames.append(existing[['Open','High','Low','Close','Volume']])
    for src in source_tickers:
        if existing is not None and not existing.empty:
            data = fetch_new_data(src, start_date=existing.index[-1]-timedelta(days=7))
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
    print(f"  ✅ {target_ticker}: {len(full_data)} 条")
    return full_data


def prepare_json_data(sp_idx, ndx_idx, etf_map):
    print("📦 对齐数据...")
    all_dates = sp_idx.index.union(ndx_idx.index).sort_values()
    df = pd.DataFrame(index=all_dates)
    df['date'] = df.index.strftime('%Y-%m-%d')
    for prefix, src in [('SP', sp_idx), ('NDX', ndx_idx)]:
        df[f'{prefix}_Close'] = src['Close'].reindex(df.index).round(2)
        df[f'{prefix}_MA200'] = src['MA200'].reindex(df.index).round(2)
        df[f'{prefix}_RSI_10'] = src['RSI_10'].reindex(df.index).round(2)
    for t, d in etf_map.items():
        df[t.replace('^', '')] = d['Close'].reindex(df.index).round(3)
    df = df[df.index >= '1990-01-01'].dropna(subset=['SP_Close'], how='all')
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient='records')


def generate_interactive_html(data_json, usd_hkd):
    json_str = json.dumps(data_json)
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>S&P 500 & Nasdaq 100 定投及增益策略</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{{--bg:#1a1a2e;--card-bg:#232336;--accent:#00d4ff;--bull:#00ff88;--bear:#ff6b6b;--text:#e0e0e0;--input-bg:#2a2a40;--input-border:#444;--warn:#ffaa00}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);padding:20px;font-size:14px}}
.container{{max-width:1400px;margin:0 auto}}
.controls{{display:flex;flex-wrap:wrap;gap:15px;background:rgba(255,255,255,0.05);padding:20px;border-radius:12px;margin-bottom:20px;align-items:flex-end;border:1px solid rgba(255,255,255,0.1)}}
.control-group{{display:flex;flex-direction:column;gap:6px}}
.control-group label{{font-size:.9em;color:#aaa;font-weight:500}}
.checkbox-group{{display:flex;align-items:center;gap:15px;background:var(--input-bg);padding:8px 12px;border:1px solid var(--input-border);border-radius:6px;height:38px}}
.checkbox-group label{{cursor:pointer;color:#fff;display:flex;align-items:center;gap:5px;margin:0}}
input[type=checkbox]{{accent-color:var(--accent);width:16px;height:16px;margin:0;min-width:auto}}
select,input[type=number],input[type=date]{{background:var(--input-bg);color:#fff;border:1px solid var(--input-border);padding:8px 12px;border-radius:6px;outline:none;font-size:14px;min-width:90px}}
select:focus,input:focus{{border-color:var(--accent)}}
optgroup{{background:#1e1e30;color:#aaa;font-style:normal}}
button{{background:linear-gradient(135deg,#00d4ff,#0077ff);color:#fff;border:none;padding:9px 25px;border-radius:6px;font-weight:bold;cursor:pointer;box-shadow:0 4px 10px rgba(0,119,255,.3);transition:.2s}}
button:hover{{opacity:.9;transform:translateY(-2px)}}
.btn-sm{{padding:5px 14px;font-size:.85em;box-shadow:none;background:linear-gradient(135deg,#555,#333)}}
.btn-sm:hover{{background:linear-gradient(135deg,#00d4ff,#0077ff)}}
.strategy-selector{{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(0,119,255,.08));padding:12px 20px;border-radius:12px;margin-bottom:15px;border:1px solid rgba(0,212,255,.3)}}
.strategy-selector label{{font-weight:bold;color:var(--accent);font-size:1.1em}}
.strategy-selector select{{font-size:1.05em;padding:8px 16px;font-weight:bold}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:20px}}
.card{{background:var(--card-bg);border-radius:12px;padding:15px;border:1px solid rgba(255,255,255,.05);box-shadow:0 4px 6px rgba(0,0,0,.1)}}
.card h3{{color:#888;font-size:.85em;margin-bottom:8px;text-transform:uppercase}}
.card .val{{font-size:1.5em;font-weight:bold;color:#fff}}
.card .sub{{font-size:.8em;color:#888;margin-top:4px}}
.action-box{{background:linear-gradient(135deg,rgba(0,212,255,.1),rgba(0,212,255,.02));border:1px solid rgba(0,212,255,.3);border-radius:12px;padding:15px;text-align:center;margin-bottom:20px;display:flex;justify-content:space-around;align-items:center;flex-wrap:wrap}}
.action-item{{margin:10px}}.action-val{{font-size:1.8em;font-weight:bold;margin:5px 0}}
.pk-box{{background:linear-gradient(135deg,rgba(255,170,0,.08),rgba(255,170,0,.02));border:1px solid rgba(255,170,0,.3);border-radius:12px;padding:15px;text-align:center;margin-bottom:20px;display:none;justify-content:space-around;align-items:center;flex-wrap:wrap}}
.pk-box.active{{display:flex}}.pk-item{{margin:10px;min-width:150px}}.pk-label{{font-size:.85em;color:#aaa;margin-bottom:4px}}.pk-val{{font-size:1.4em;font-weight:bold}}
.chart-container{{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:20px}}
.chart-box{{background:var(--card-bg);padding:15px;border-radius:12px;border:1px solid rgba(255,255,255,.05)}}
.chart-title{{font-size:1.1em;color:var(--accent);margin-bottom:10px;font-weight:bold}}
.legend-bar{{display:flex;justify-content:center;align-items:center;gap:20px;flex-wrap:wrap;background:rgba(0,0,0,.2);padding:8px;border-radius:6px;margin-bottom:10px;font-size:.9em;color:#ccc}}
.legend-item{{display:flex;align-items:center;gap:6px}}
table{{width:100%;border-collapse:collapse;font-size:.9em}}
th,td{{padding:10px;text-align:right;border-bottom:1px solid rgba(255,255,255,.1)}}
th{{background:rgba(0,0,0,.2);position:sticky;top:0;color:#aaa}}
td:first-child,th:first-child{{text-align:left}}
.scroll-table{{max-height:400px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#444 #222}}
.c-bull{{color:var(--bull)}}.c-bear{{color:var(--bear)}}.c-warn{{color:#ffaa00}}
tr.highlight{{background:rgba(255,170,0,.1)}}
.ipo-hint{{font-size:.75em;color:#666;margin-top:4px;line-height:1.4}}
.heat-table{{margin-bottom:20px}}
.heat-pos{{color:var(--bull);font-weight:bold}}.heat-neg{{color:var(--bear);font-weight:bold}}
#logTable td:last-child,#logTable th:last-child{{text-align:left}}
@media(max-width:768px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="container">
<div style="text-align:center;margin-bottom:20px">
<h2 style="color:var(--accent)">📈 S&P 500 & Nasdaq 100 定投及增益策略</h2>
<div style="font-size:.9em;color:#888">汇率: 1 USD = {usd_hkd:.4f} HKD | MA200牛熊分界 | RSI(10) | XIRR年化</div>
</div>
<div class="strategy-selector"><label>🎯 投资策略:</label>
<select id="strategyMode" onchange="onStrategyChange()">
<option value="dca">📅 定投策略</option><option value="bottom">🎯 抄底策略</option>
</select></div>
<div class="controls">
<div class="control-group"><label>开始时间</label>
<div style="display:flex;gap:5px">
<select id="periodSelect" style="width:110px" onchange="onPeriodChange()">
<option value="0">1993-2000</option><option value="1">2001-2010</option><option value="2">2011-2020</option><option value="3" selected>2021-至今</option>
</select>
<select id="startYear" style="width:70px" onchange="onDatePartChange()"></select>
<select id="startMonth" style="width:60px" onchange="onDatePartChange()">
<option value="1">1月</option><option value="2">2月</option><option value="3">3月</option><option value="4">4月</option><option value="5">5月</option><option value="6">6月</option>
<option value="7">7月</option><option value="8">8月</option><option value="9">9月</option><option value="10">10月</option><option value="11">11月</option><option value="12">12月</option>
</select>
<select id="startDay" style="width:60px" onchange="onDatePartChange()"></select>
</div></div>
<div class="control-group"><label style="color:var(--accent)">1. 投资标的①</label>
<select id="tickerSelect" onchange="onTickerChange(1)">
<optgroup label="── S&P 500 ──"><option value="SPY">SPY (1993)</option><option value="IVV">IVV (2000)</option><option value="SPYM">SPYM (2005)</option><option value="VOO">VOO (2010)</option><option value="SSO">SSO 2倍 (2006)</option><option value="UPRO">UPRO 3倍 (2009)</option></optgroup>
<optgroup label="── Nasdaq 100 ──"><option value="QQQ">QQQ (1999)</option><option value="QQQM">QQQM (2020)</option><option value="QLD">QLD 2倍 (2006)</option><option value="TQQQ">TQQQ 3倍 (2010)</option></optgroup>
</select><div class="ipo-hint" id="ipoHint"></div></div>
<div class="control-group"><label style="color:#ffaa00">⚔️ PK对比②</label>
<select id="tickerSelect2" onchange="onTickerChange(2)">
<option value="NONE">无 (不对比)</option>
<optgroup label="── S&P 500 ──"><option value="SPY">SPY</option><option value="IVV">IVV</option><option value="SPYM">SPYM</option><option value="VOO">VOO</option><option value="SSO">SSO 2倍</option><option value="UPRO">UPRO 3倍</option></optgroup>
<optgroup label="── Nasdaq 100 ──"><option value="QQQ">QQQ</option><option value="QQQM">QQQM</option><option value="QLD">QLD 2倍</option><option value="TQQQ">TQQQ 3倍</option></optgroup>
</select><div class="ipo-hint" id="ipoHint2"></div></div>
<div class="control-group"><label style="color:var(--bull)">3. 策略配置</label>
<div class="checkbox-group">
<label id="lblChkBase"><input type="checkbox" id="chkBase" checked onchange="runStrategy()"> <span id="chkBaseText">基础定投</span></label>
<label id="lblChkBear"><input type="checkbox" id="chkBear" checked onchange="runStrategy()"> <span id="chkBearText">熊市加倍</span></label>
</div></div>
<div class="control-group"><label>基准金额 (HKD)</label><input type="number" id="baseInvest" value="2000" step="500"></div>
<div class="control-group"><label id="lblMulti">加倍倍数</label>
<select id="bearMulti" onchange="runStrategy()"><option value="2" selected>2倍</option><option value="3">3倍</option><option value="4">4倍</option><option value="5">5倍</option><option value="6">6倍</option><option value="7">7倍</option><option value="8">8倍</option><option value="9">9倍</option><option value="10">10倍</option></select></div>
<div class="control-group"><label>查看截至</label><input type="date" id="endDate"></div>
<div class="control-group"><label>&nbsp;</label><button onclick="runStrategy()">🔄 重新计算</button></div>
</div>
<div class="action-box">
<div class="action-item"><div>当前持仓 (<span id="lblTicker1Tag">①</span>)</div><div id="statusPos" class="action-val" style="color:#fff">--</div><div id="statusPosDetail" style="font-size:.75em;opacity:.7">--</div></div>
<div class="action-item"><div>下期建议 (<span id="lblTicker">SPY</span>)</div><div id="statusNextInvest" class="action-val">--</div><div id="statusMarket" style="font-size:.8em;opacity:.8">--</div></div>
<div class="action-item"><div>总收益率 ①</div><div id="statusReturn" class="action-val">--</div></div>
<div class="action-item"><div>年化收益率 ①</div><div id="statusAnnual" class="action-val">--</div><div id="statusAnnualNote" style="font-size:.75em;opacity:.7">XIRR | N=<span id="annN">0</span>笔 跨<span id="annDays">0</span>天</div></div>
</div>
<div class="pk-box" id="pkBox">
<div class="pk-item"><div class="pk-label">⚔️ PK对比</div><div class="pk-val" style="color:var(--accent)">① vs ②</div></div>
<div class="pk-item"><div class="pk-label" id="pkLabel1">①</div><div class="pk-val c-bull" id="pkReturn1">--</div><div style="font-size:.75em;color:#888">年化: <span id="pkAnnual1">--</span></div></div>
<div class="pk-item"><div class="pk-label" id="pkLabel2">②</div><div class="pk-val c-warn" id="pkReturn2">--</div><div style="font-size:.75em;color:#888">年化: <span id="pkAnnual2">--</span></div></div>
<div class="pk-item"><div class="pk-label">本金差</div><div class="pk-val" id="pkInvDiff" style="color:#ccc">--</div></div>
<div class="pk-item"><div class="pk-label">市值差</div><div class="pk-val" id="pkValDiff" style="color:#ccc">--</div></div>
</div>
<div class="grid">
<div class="card"><h3 id="idxCardTitle">标普500指数</h3><div class="val" id="valIdx">--</div><div class="sub" id="valDate">--</div></div>
<div class="card"><h3>日线 MA200</h3><div class="val" id="valMa200">--</div><div class="sub">牛熊分界 (<span id="lblIdxFamily">S&P</span>)</div></div>
<div class="card"><h3>RSI (10)</h3><div class="val" id="valRsi">--</div><div class="sub" id="valRsiSig">--</div></div>
<div class="card"><h3>份额详情 (①)</h3><div class="sub"><span id="lblShareTicker">SPY</span>: <span id="shareMain" style="color:#fff">0</span> 股</div></div>
</div>
<div class="grid">
<div class="card"><h3>累计本金 ①</h3><div class="val" id="valTotalInv">--</div><div class="sub">HKD</div></div>
<div class="card"><h3>总市值 ①</h3><div class="val" id="valTotalVal">--</div><div class="sub">HKD</div></div>
<div class="card"><h3 id="lblCount1Title">基础定投期数</h3><div class="val" id="valBaseCount">0</div><div class="sub" id="lblCount1Sub">牛市单倍</div></div>
<div class="card"><h3 id="lblCount2Title">熊市倍投期数</h3><div class="val" id="valBearCount">0</div><div class="sub" id="lblCount2Sub">熊市N倍</div></div>
</div>
<div class="grid">
<div class="card"><h3>最大回撤</h3><div class="val c-bear" id="valMDD">--</div><div class="sub">Max Drawdown</div></div>
<div class="card"><h3>夏普比率</h3><div class="val" id="valSharpe">--</div><div class="sub">Sharpe (Rf=4%)</div></div>
<div class="card"><h3>索提诺比率</h3><div class="val" id="valSortino">--</div><div class="sub">Sortino (下行风险)</div></div>
<div class="card"><h3>胜率</h3><div class="val" id="valWinRate">--</div><div class="sub">Win Rate (日度盈利占比)</div></div>
</div>
<div class="chart-box heat-table"><div class="chart-title">📅 年度收益热力表 ①</div>
<div class="scroll-table" style="max-height:300px"><table id="yearTable">
<thead><tr><th style="text-align:left">年份</th><th>年初市值</th><th>当年投入</th><th>年末市值</th><th>当年盈亏</th><th>盈亏比例</th><th>累计本金</th><th>累计收益率</th></tr></thead>
<tbody></tbody></table></div></div>
<div class="chart-container">
<div class="chart-box"><div class="chart-title">💰 账户总市值趋势 (HKD) <span id="chartPkHint" style="font-size:.8em;color:#ffaa00"></span></div>
<canvas id="mainChart" height="280" style="max-height:280px"></canvas></div>
<div class="chart-box"><div class="chart-title" id="techChartTitle">📊 市场信号</div>
<div class="legend-bar">
<div class="legend-item"><span style="color:red">▲</span> RSI&lt;30 (超卖)</div><div style="color:#555">|</div>
<div class="legend-item"><span style="color:#ff00ff">▲</span> RSI&lt;20 (极端超卖)</div><div style="color:#555">|</div>
<div class="legend-item"><span style="color:#00dd00">▼</span> RSI&gt;80 (超买)</div><div style="color:#555">|</div>
<div class="legend-item"><span style="color:#0070FF">▼</span> RSI&gt;85 (极端超买)</div>
</div>
<canvas id="techChart" height="300" style="max-height:300px"></canvas></div></div>
<div class="chart-box scroll-table">
<div class="chart-title" style="display:flex;justify-content:space-between;align-items:center">
<span>📜 交易流水 ① (倒序)</span>
<button class="btn-sm" onclick="downloadCSV()">⬇️ 下载CSV</button></div>
<table id="logTable"><thead><tr>
<th style="text-align:left">日期</th><th>指数</th><th>RSI</th><th>牛熊</th><th>投入(HKD)</th><th>累计本金</th><th>总市值</th>
<th><span id="thShareTicker">SPY</span> 份额</th><th style="text-align:left">操作</th>
</tr></thead><tbody></tbody></table></div>
</div>
<script>
const RAW_DATA={json_str};
const USD_HKD={usd_hkd};
const RISK_FREE_ANNUAL=0.04;
const NDX_SET=new Set(['QQQ','QQQM','QLD','TQQQ']);
function getFamily(t){{return NDX_SET.has(t)?'NDX':'SP';}}
function getFamilyLabel(t){{return getFamily(t)==='NDX'?'Nasdaq 100':'S&P 500';}}
function getIdxData(row,fam){{if(fam==='NDX')return{{close:row.NDX_Close,ma200:row.NDX_MA200,rsi:row.NDX_RSI_10}};return{{close:row.SP_Close,ma200:row.SP_MA200,rsi:row.SP_RSI_10}};}}
const IPO_DATES={{'SPY':'1993-01-22','IVV':'2000-05-15','SPYM':'2005-11-08','VOO':'2010-09-07','SSO':'2006-06-19','UPRO':'2009-06-25','QQQ':'1999-03-10','QQQM':'2020-10-13','QLD':'2006-06-19','TQQQ':'2010-02-09'}};
const IPO_NAMES={{'SPY':'SPY','IVV':'IVV','SPYM':'SPYM','VOO':'VOO','SSO':'SSO(2x)','UPRO':'UPRO(3x)','QQQ':'QQQ','QQQM':'QQQM','QLD':'QLD(2x)','TQQQ':'TQQQ(3x)'}};
const PERIODS=[{{label:'1993-2000',start:1993,end:2000}},{{label:'2001-2010',start:2001,end:2010}},{{label:'2011-2020',start:2011,end:2020}},{{label:'2021-至今',start:2021,end:new Date().getFullYear()}}];
let currentLogs=[],currentTicker='SPY';
function initUI(){{const ds=document.getElementById('startDay');for(let d=1;d<=31;d++){{const o=document.createElement('option');o.value=d;o.text=d+'日';ds.appendChild(o);}}updateYearOptions(2021);if(RAW_DATA.length>0)document.getElementById('endDate').value=RAW_DATA[RAW_DATA.length-1].date;updateIpoHint();onStrategyChange();runStrategy();}}
function onStrategyChange(){{const m=document.getElementById('strategyMode').value;if(m==='dca'){{document.getElementById('chkBaseText').innerText='基础定投';document.getElementById('chkBearText').innerText='熊市加倍';document.getElementById('lblCount1Title').innerText='基础定投期数';document.getElementById('lblCount1Sub').innerText='牛市单倍';document.getElementById('lblCount2Title').innerText='熊市倍投期数';document.getElementById('lblCount2Sub').innerText='熊市N倍';document.getElementById('lblMulti').innerText='熊市倍数';}}else{{document.getElementById('chkBaseText').innerText='基础抄底';document.getElementById('chkBearText').innerText='超熊加倍';document.getElementById('lblCount1Title').innerText='抄底投资天数';document.getElementById('lblCount1Sub').innerText='RSI<30 且 ≥20';document.getElementById('lblCount2Title').innerText='超熊加倍次数';document.getElementById('lblCount2Sub').innerText='RSI<20 N倍';document.getElementById('lblMulti').innerText='超熊倍数';}}runStrategy();}}
function onPeriodChange(){{updateYearOptions(null);onDatePartChange();}}
function updateYearOptions(dy){{const p=PERIODS[parseInt(document.getElementById('periodSelect').value)];const s=document.getElementById('startYear');s.innerHTML='';for(let y=p.start;y<=p.end;y++){{const o=document.createElement('option');o.value=y;o.text=y;s.appendChild(o);}}if(dy&&dy>=p.start&&dy<=p.end)s.value=dy;}}
function onDatePartChange(){{validateTickerVsDate(false,'tickerSelect');validateTickerVsDate(false,'tickerSelect2');runStrategy();}}
function onTickerChange(w){{validateTickerVsDate(true,w===1?'tickerSelect':'tickerSelect2');updateIpoHint();runStrategy();}}
function updateIpoHint(){{const t1=document.getElementById('tickerSelect').value;document.getElementById('ipoHint').innerHTML='['+getFamilyLabel(t1)+'] '+IPO_NAMES[t1]+' 上市: '+IPO_DATES[t1];const t2=document.getElementById('tickerSelect2').value;document.getElementById('ipoHint2').innerHTML=t2!=='NONE'?'['+getFamilyLabel(t2)+'] '+IPO_NAMES[t2]+' 上市: '+IPO_DATES[t2]:'';}}
function validateTickerVsDate(fromTicker,selId){{const sel=document.getElementById(selId),t=sel.value;if(t==='NONE')return;const y=parseInt(document.getElementById('startYear').value),m=parseInt(document.getElementById('startMonth').value);const d0=parseInt(document.getElementById('startDay').value),dim=new Date(y,m,0).getDate(),d=Math.min(d0,dim);const s=y+'-'+String(m).padStart(2,'0')+'-'+String(d).padStart(2,'0');if(s<IPO_DATES[t]){{if(fromTicker){{if(selId==='tickerSelect'){{alert(IPO_NAMES[t]+' 上市: '+IPO_DATES[t]+'，已切换为SPY');sel.value='SPY';}}else{{alert(IPO_NAMES[t]+' 上市: '+IPO_DATES[t]+'，已取消对比');sel.value='NONE';}}}}else{{if(selId==='tickerSelect'&&t!=='SPY')sel.value='SPY';if(selId==='tickerSelect2'&&t!=='NONE')sel.value='NONE';}}updateIpoHint();}}}}
function buildDcaDateStr(y,m,pd){{const dim=new Date(y,m,0).getDate();return y+'-'+String(m).padStart(2,'0')+'-'+String(Math.min(pd,dim)).padStart(2,'0');}}
function advanceMonth(y,m){{m++;if(m>12){{m=1;y++;}}return[y,m];}}
function calcXIRR(cf){{if(!cf||cf.length<2)return 0;if(!cf.some(c=>c.amount<0)||!cf.some(c=>c.amount>0))return 0;function pD(s){{return new Date(s+'T00:00:00');}}const d0=pD(cf[0].date);function yF(d){{return(pD(d).getTime()-d0.getTime())/(365.25*86400000);}}const tD=(pD(cf[cf.length-1].date).getTime()-d0.getTime())/86400000;if(tD<30){{const ti=cf.filter(c=>c.amount<0).reduce((s,c)=>s-c.amount,0);if(ti<=0||tD<=0)return 0;return Math.pow(1+(cf[cf.length-1].amount-ti)/ti,365.25/tD)-1;}}function npv(r){{let s=0;for(const c of cf){{const b=1+r;if(b<=0)return 1e15;s+=c.amount/Math.pow(b,yF(c.date));}}return s;}}function dnpv(r){{let s=0;for(const c of cf){{const t=yF(c.date),b=1+r;if(b<=0)return-1e15;s-=t*c.amount/Math.pow(b,t+1);}}return s;}}let bestR=null,bestN=Infinity;for(const g of[.1,.2,0,-.1,.5,1]){{let r=g;for(let i=0;i<500;i++){{const f=npv(r),df=dnpv(r);if(Math.abs(df)<1e-15)break;const rN=r-f/df;if(Math.abs(rN-r)<1e-12){{r=rN;break;}}r=rN;if(r<-.99)r=-.5;if(r>100)r=5;}}const a=Math.abs(npv(r));if(a<bestN&&r>-.99&&r<100&&isFinite(r)&&!isNaN(r)){{bestN=a;bestR=r;}}}}return bestR!==null?bestR:0;}}
let valChart=null,techChart=null;
function executeStrategy(tk,startDateStr,viewEndDate,baseInvest,bearMulti,preferredDay,useBase,useBear,startYear,startMonth,strategyMode){{const family=getFamily(tk);let nDY=startYear,nDM=startMonth,nDS=buildDcaDateStr(nDY,nDM,preferredDay);let S={{total_invested:0,shares:0,base_count:0,bear_count:0}};let logs=[],cashFlows=[];let cDates=[],cVals=[],cIdx=[],cMAs=[],cRSIs=[];let cSig30=[],cSig20=[],cSig80=[],cSig85=[];let prevRsi=null;for(let i=0;i<RAW_DATA.length;i++){{const row=RAW_DATA[i],idx=getIdxData(row,family),rsi=idx.rsi;if(row.date<startDateStr){{if(rsi!=null)prevRsi=rsi;continue;}}if(row.date>viewEndDate)break;const idxP=idx.close,ma200=idx.ma200,mainP=row[tk];if(!idxP||!ma200||rsi==null||!mainP){{cDates.push(row.date);cVals.push(null);cIdx.push(null);cMAs.push(null);cRSIs.push(null);cSig30.push(null);cSig20.push(null);cSig80.push(null);cSig85.push(null);if(rsi!=null)prevRsi=rsi;continue;}}const isBull=idxP>ma200;let action='',note='',investAmt=0;if(strategyMode==='dca'){{if(useBase){{let dcaDay=0,dcaN=[];while(row.date>=nDS){{let inv=baseInvest;if(useBear&&!isBull){{inv=baseInvest*bearMulti;dcaN.push("熊市"+bearMulti+"倍");S.bear_count++;}}else{{dcaN.push("基础定投");S.base_count++;}}S.total_invested+=inv;dcaDay+=inv;S.shares+=(inv/USD_HKD)/mainP;cashFlows.push({{date:row.date,amount:-inv}});[nDY,nDM]=advanceMonth(nDY,nDM);nDS=buildDcaDateStr(nDY,nDM,preferredDay);}}if(dcaDay>0){{action="定投";investAmt=dcaDay;note=dcaN.join(" + ");}}}}}}else if(strategyMode==='bottom'){{if(useBase&&rsi<30){{let inv=baseInvest;if(useBear&&rsi<20){{inv=baseInvest*bearMulti;S.bear_count++;action="抄底+超熊"+bearMulti+"倍";note="RSI="+rsi.toFixed(1)+"<20";}}else{{S.base_count++;action="抄底";note="RSI="+rsi.toFixed(1)+"<30";}}S.total_invested+=inv;S.shares+=(inv/USD_HKD)/mainP;investAmt=inv;cashFlows.push({{date:row.date,amount:-inv}});}}}}const totalValHkd=S.shares*mainP*USD_HKD;const isLast=(i===RAW_DATA.length-1)||(row.date===viewEndDate);if(action!==''||isLast){{logs.push({{date:row.date,idx:idxP,rsi:rsi,isBull:isBull,invest:investAmt,totalInv:S.total_invested,totalVal:totalValHkd,shares:S.shares,action:action,note:note,baseCount:S.base_count,bearCount:S.bear_count}});}}cDates.push(row.date);cVals.push(totalValHkd);cIdx.push(idxP);cMAs.push(ma200);cRSIs.push(rsi);let sig30=null,sig20=null,sig80=null,sig85=null;if(prevRsi!==null){{if(rsi<30&&prevRsi>=30)sig30=rsi;if(rsi<20&&prevRsi>=20)sig20=rsi;if(rsi>80&&prevRsi<=80)sig80=rsi;if(rsi>85&&prevRsi<=85)sig85=rsi;}}cSig30.push(sig30);cSig20.push(sig20);cSig80.push(sig80);cSig85.push(sig85);prevRsi=rsi;}}if(logs.length>0&&cashFlows.length>0)cashFlows.push({{date:logs[logs.length-1].date,amount:logs[logs.length-1].totalVal}});return{{logs,cashFlows,chartDates:cDates,chartVals:cVals,chartIdx:cIdx,chartMAs:cMAs,chartRSIs:cRSIs,cSig30,cSig20,cSig80,cSig85}};}}
function calcRiskMetrics(v){{const vals=v.filter(x=>x!==null&&x>0);if(vals.length<20)return{{mdd:0,sharpe:0,sortino:0,winRate:0}};let peak=vals[0],maxDD=0;for(let i=1;i<vals.length;i++){{if(vals[i]>peak)peak=vals[i];const dd=(peak-vals[i])/peak;if(dd>maxDD)maxDD=dd;}}const rets=[];for(let i=1;i<vals.length;i++)if(vals[i-1]>0)rets.push(vals[i]/vals[i-1]-1);if(rets.length<2)return{{mdd:maxDD,sharpe:0,sortino:0,winRate:0}};const m=rets.reduce((a,b)=>a+b,0)/rets.length,rf=RISK_FREE_ANNUAL/252;const std=Math.sqrt(rets.reduce((a,b)=>a+(b-m)**2,0)/(rets.length-1));const dR=rets.filter(r=>r<rf),dStd=Math.sqrt(dR.length>1?dR.reduce((a,b)=>a+(b-rf)**2,0)/(dR.length-1):0);return{{mdd:maxDD,sharpe:std>0?(m-rf)/std*Math.sqrt(252):0,sortino:dStd>0?(m-rf)/dStd*Math.sqrt(252):0,winRate:rets.length>0?(rets.filter(r=>r>0).length/rets.length*100):0}};}}
function buildYearlyTable(cD,cV,logs){{const tbody=document.querySelector('#yearTable tbody');tbody.innerHTML='';if(!cD||!cD.length)return;const inv={{}};logs.forEach(l=>{{if(l.invest>0)inv[l.date]=(inv[l.date]||0)+l.invest;}});const yrs={{}};for(let i=0;i<cD.length;i++){{const v=cV[i];if(v===null)continue;const y=cD[i].substring(0,4);if(!yrs[y])yrs[y]={{s:null,e:0,i:0}};if(yrs[y].s===null)yrs[y].s=v;yrs[y].e=v;if(inv[cD[i]])yrs[y].i+=inv[cD[i]];}}let cum=0,prev=0;Object.keys(yrs).sort().forEach(y=>{{const d=yrs[y],sv=prev>0?prev:d.s,pl=d.e-sv-d.i,b=sv+d.i,pct=b>0?(pl/b*100):0;cum+=d.i;const cr=cum>0?((d.e-cum)/cum*100):0;const cl=pl>=0?'heat-pos':'heat-neg';const tr=document.createElement('tr');tr.innerHTML='<td style="text-align:left"><strong>'+y+'</strong></td><td>'+Math.round(sv).toLocaleString()+'</td><td>'+Math.round(d.i).toLocaleString()+'</td><td>'+Math.round(d.e).toLocaleString()+'</td><td class="'+cl+'">'+(pl>=0?'+':'')+Math.round(pl).toLocaleString()+'</td><td class="'+cl+'">'+(pct>=0?'+':'')+pct.toFixed(1)+'%</td><td>'+Math.round(cum).toLocaleString()+'</td><td class="'+(cr>=0?'heat-pos':'heat-neg')+'">'+(cr>=0?'+':'')+cr.toFixed(1)+'%</td>';tbody.appendChild(tr);prev=d.e;}});}}
function downloadCSV(){{if(!currentLogs||!currentLogs.length){{alert('暂无数据');return;}}const h=['日期','指数','RSI','牛熊','投入(HKD)','累计本金','总市值',currentTicker+'份额','操作','备注'];const rows=[h.join(',')];currentLogs.forEach(l=>{{rows.push([l.date,l.idx.toFixed(2),l.rsi.toFixed(1),l.isBull?'牛市':'熊市',l.invest>0?l.invest.toFixed(0):'',l.totalInv.toFixed(0),l.totalVal.toFixed(0),l.shares.toFixed(4),'"'+l.action.replace(/"/g,'""')+'"','"'+l.note.replace(/"/g,'""')+'"'].join(','));}});const blob=new Blob(['\\uFEFF'+rows.join('\\n')],{{type:'text/csv;charset=utf-8;'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='交易流水_'+currentTicker+'_'+new Date().toISOString().slice(0,10)+'.csv';a.click();}}
function runStrategy(){{const startYear=parseInt(document.getElementById('startYear').value),startMonth=parseInt(document.getElementById('startMonth').value);const startDayReq=parseInt(document.getElementById('startDay').value),baseInvest=parseFloat(document.getElementById('baseInvest').value);const bearMulti=parseInt(document.getElementById('bearMulti').value);let viewEndDate=document.getElementById('endDate').value;const t1=document.getElementById('tickerSelect').value,t2=document.getElementById('tickerSelect2').value;const mode=document.getElementById('strategyMode').value;const useBase=document.getElementById('chkBase').checked,useBear=document.getElementById('chkBear').checked;document.getElementById('lblTicker').innerText=t1;document.getElementById('lblTicker1Tag').innerText='① '+t1;document.getElementById('lblShareTicker').innerText=t1;document.getElementById('thShareTicker').innerText=t1;document.getElementById('idxCardTitle').innerText=getFamily(t1)==='NDX'?'纳斯达克100指数':'标普500指数';document.getElementById('lblIdxFamily').innerText=getFamilyLabel(t1);document.getElementById('techChartTitle').innerText='📊 市场信号 — '+getFamilyLabel(t1);const dim=new Date(startYear,startMonth,0).getDate();const startDateStr=startYear+'-'+String(startMonth).padStart(2,'0')+'-'+String(Math.min(startDayReq,dim)).padStart(2,'0');if(!viewEndDate&&RAW_DATA.length>0)viewEndDate=RAW_DATA[RAW_DATA.length-1].date;const r1=executeStrategy(t1,startDateStr,viewEndDate,baseInvest,bearMulti,startDayReq,useBase,useBear,startYear,startMonth,mode);let r2=null;const hasPK=t2!=='NONE';if(hasPK)r2=executeStrategy(t2,startDateStr,viewEndDate,baseInvest,bearMulti,startDayReq,useBase,useBear,startYear,startMonth,mode);currentLogs=r1.logs;currentTicker=t1;updateStatusCards(r1,baseInvest,bearMulti,t1,useBase,useBear,mode);updateTable(r1.logs,t1);updateCharts(r1,r2,t1,t2);updatePKBox(r1,r2,t1,t2,hasPK);buildYearlyTable(r1.chartDates,r1.chartVals,r1.logs);const risk=calcRiskMetrics(r1.chartVals);document.getElementById('valMDD').innerText='-'+(risk.mdd*100).toFixed(2)+'%';const shEl=document.getElementById('valSharpe');shEl.innerText=risk.sharpe.toFixed(2);shEl.style.color=risk.sharpe>=1?'var(--bull)':(risk.sharpe>=0?'var(--warn)':'var(--bear)');const soEl=document.getElementById('valSortino');soEl.innerText=risk.sortino.toFixed(2);soEl.style.color=risk.sortino>=1.5?'var(--bull)':(risk.sortino>=0?'var(--warn)':'var(--bear)');const wrEl=document.getElementById('valWinRate');wrEl.innerText=risk.winRate.toFixed(1)+'%';wrEl.style.color=risk.winRate>=55?'var(--bull)':(risk.winRate>=45?'var(--warn)':'var(--bear)');}}
function updateStatusCards(r1,baseInvest,bearMulti,tk,useBase,useBear,mode){{const logs=r1.logs,last=logs[logs.length-1];if(!last)return;const fam=getFamily(tk),rawRow=RAW_DATA.find(r=>r.date===last.date);const ma200=rawRow?(fam==='NDX'?rawRow.NDX_MA200:rawRow.SP_MA200):0;const isBull=last.idx>ma200;document.getElementById('valIdx').innerText=last.idx.toFixed(2);document.getElementById('valDate').innerText=last.date;document.getElementById('valMa200').innerText=ma200.toFixed(2);const rE=document.getElementById('valRsi');rE.innerText=last.rsi.toFixed(1);rE.style.color=last.rsi<30?'var(--bear)':(last.rsi>70?'var(--warn)':'var(--bull)');let sig='正常';if(last.rsi<30)sig=isBull?'⚠️ 牛市急跌':'📉 熊市超卖';if(last.rsi>70)sig='🔵 超买警戒';if(last.rsi>80)sig='🟢 极度超买';if(last.rsi>85)sig='🔥 超极端';document.getElementById('valRsiSig').innerText=sig;document.getElementById('valTotalInv').innerText=Math.round(last.totalInv).toLocaleString();document.getElementById('valTotalVal').innerText=Math.round(last.totalVal).toLocaleString();document.getElementById('valBaseCount').innerText=last.baseCount;document.getElementById('valBearCount').innerText=last.bearCount;document.getElementById('shareMain').innerText=last.shares.toFixed(4);const ret=last.totalInv>0?((last.totalVal-last.totalInv)/last.totalInv*100):0;document.getElementById('statusReturn').innerText=(ret>0?'+':'')+ret.toFixed(2)+'%';document.getElementById('statusReturn').className='action-val '+(ret>=0?'c-bull':'c-bear');let ann=0;if(r1.cashFlows&&r1.cashFlows.length>=2)ann=calcXIRR(r1.cashFlows)*100;const aE=document.getElementById('statusAnnual');aE.innerText=(ann>0?'+':'')+ann.toFixed(2)+'%';aE.className='action-val '+(ann>=0?'c-bull':'c-bear');document.getElementById('annN').innerText=last.baseCount+last.bearCount;if(r1.cashFlows&&r1.cashFlows.length>=2){{const d0=new Date(r1.cashFlows[0].date+'T00:00:00'),d1=new Date(r1.cashFlows[r1.cashFlows.length-1].date+'T00:00:00');document.getElementById('annDays').innerText=Math.round((d1-d0)/86400000);}}else document.getElementById('annDays').innerText='0';document.getElementById('statusPos').innerHTML='<span style="color:var(--bull);font-weight:bold">🟢 '+tk+'</span>';document.getElementById('statusPosDetail').innerText=tk+' '+last.shares.toFixed(4)+' 股';if(mode==='dca'){{let nI=0;if(useBase)nI=(useBear&&!isBull)?baseInvest*bearMulti:baseInvest;document.getElementById('statusNextInvest').innerText='HK$ '+nI.toLocaleString();}}else{{if(last.rsi<20&&useBear)document.getElementById('statusNextInvest').innerText='HK$ '+(baseInvest*bearMulti).toLocaleString()+'/天 (超熊'+bearMulti+'x)';else if(last.rsi<30)document.getElementById('statusNextInvest').innerText='HK$ '+baseInvest.toLocaleString()+'/天';else document.getElementById('statusNextInvest').innerText='等待 RSI<30';}}document.getElementById('statusMarket').innerHTML='<span class="'+(isBull?'c-bull':'c-bear')+'">'+(isBull?'🐂 牛市':'🐻 熊市')+' ('+getFamilyLabel(tk)+')</span>';}}
function updatePKBox(r1,r2,t1,t2,hasPK){{const box=document.getElementById('pkBox');if(!hasPK||!r2){{box.classList.remove('active');document.getElementById('chartPkHint').innerText='';return;}}box.classList.add('active');document.getElementById('chartPkHint').innerText='  — ① '+t1+'('+getFamilyLabel(t1)+') vs ② '+t2+'('+getFamilyLabel(t2)+')';const l1=r1.logs[r1.logs.length-1],l2=r2.logs[r2.logs.length-1];if(!l1||!l2)return;const ret1=l1.totalInv>0?((l1.totalVal-l1.totalInv)/l1.totalInv*100):0,ret2=l2.totalInv>0?((l2.totalVal-l2.totalInv)/l2.totalInv*100):0;const ann1=(r1.cashFlows&&r1.cashFlows.length>=2)?calcXIRR(r1.cashFlows)*100:0,ann2=(r2.cashFlows&&r2.cashFlows.length>=2)?calcXIRR(r2.cashFlows)*100:0;document.getElementById('pkLabel1').innerText='① '+t1+' ('+getFamilyLabel(t1)+')';document.getElementById('pkLabel2').innerText='② '+t2+' ('+getFamilyLabel(t2)+')';const fmt=v=>(v>0?'+':'')+v.toFixed(2)+'%';document.getElementById('pkReturn1').innerText=fmt(ret1);document.getElementById('pkReturn1').className='pk-val '+(ret1>=0?'c-bull':'c-bear');document.getElementById('pkReturn2').innerText=fmt(ret2);document.getElementById('pkReturn2').className='pk-val '+(ret2>=0?'c-bull':'c-bear');document.getElementById('pkAnnual1').innerText=fmt(ann1);document.getElementById('pkAnnual2').innerText=fmt(ann2);const invD=l1.totalInv-l2.totalInv,valD=l1.totalVal-l2.totalVal;document.getElementById('pkInvDiff').innerText=(invD>=0?'+':'')+Math.round(invD).toLocaleString()+' HKD';const vd=document.getElementById('pkValDiff');vd.innerText=(valD>=0?'+':'')+Math.round(valD).toLocaleString()+' HKD';vd.className='pk-val '+(valD>=0?'c-bull':'c-bear');}}
function updateTable(logs,tk){{const tbody=document.querySelector('#logTable tbody');tbody.innerHTML='';logs.slice().reverse().slice(0,200).forEach(l=>{{const tr=document.createElement('tr');if(l.action.includes('超熊'))tr.className='highlight';tr.innerHTML='<td style="text-align:left">'+l.date+'</td><td>'+l.idx.toFixed(2)+'</td><td class="'+(l.rsi<30?'c-bear':(l.rsi>70?'c-warn':''))+'">'+l.rsi.toFixed(1)+'</td><td class="'+(l.isBull?'c-bull':'c-bear')+'" style="text-align:center">'+(l.isBull?'🐂':'🐻')+'</td><td>'+(l.invest>0?'+'+l.invest.toLocaleString():'')+'</td><td>'+Math.round(l.totalInv).toLocaleString()+'</td><td>'+Math.round(l.totalVal).toLocaleString()+'</td><td>'+l.shares.toFixed(4)+'</td><td style="text-align:left"><strong>'+l.action+'</strong><div style="font-size:.8em;color:#aaa">'+l.note+'</div></td>';tbody.appendChild(tr);}});}}
function updateCharts(r1,r2,t1,t2){{const dates=r1.chartDates,f1=getFamilyLabel(t1);const ctxV=document.getElementById('mainChart').getContext('2d');if(valChart)valChart.destroy();const ds=[{{label:'① '+t1+' ('+f1+')',data:r1.chartVals,borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,.08)',borderWidth:2,fill:true,pointRadius:0,pointHitRadius:10}}];if(r2){{const f2=getFamilyLabel(t2),map2={{}};r2.chartDates.forEach((d,i)=>{{map2[d]=r2.chartVals[i];}});ds.push({{label:'② '+t2+' ('+f2+')',data:dates.map(d=>map2[d]!==undefined?map2[d]:null),borderColor:'#ffaa00',backgroundColor:'rgba(255,170,0,.06)',borderWidth:2,fill:true,pointRadius:0,pointHitRadius:10,borderDash:[6,3]}});}}valChart=new Chart(ctxV,{{type:'line',data:{{labels:dates,datasets:ds}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{display:!!r2,labels:{{color:'#ccc'}}}}}},scales:{{x:{{display:false}},y:{{position:'right',grid:{{color:'rgba(255,255,255,.05)'}},ticks:{{color:'#888'}}}}}}}}}});const ctxT=document.getElementById('techChart').getContext('2d');if(techChart)techChart.destroy();techChart=new Chart(ctxT,{{type:'line',data:{{labels:dates,datasets:[{{label:f1+' Index',data:r1.chartIdx,borderColor:'#00d4ff',borderWidth:1.5,yAxisID:'y',pointRadius:0}},{{label:'MA200',data:r1.chartMAs,borderColor:'#ff6b6b',borderWidth:1.5,borderDash:[5,5],yAxisID:'y',pointRadius:0}},{{label:'RSI(10)',data:r1.chartRSIs,borderColor:'#ffaa00',borderWidth:1,yAxisID:'y1',pointRadius:0}},{{label:'RSI<30',data:r1.cSig30,borderColor:'red',backgroundColor:'red',pointStyle:'triangle',rotation:0,pointRadius:7,showLine:false,yAxisID:'y1'}},{{label:'RSI<20',data:r1.cSig20,borderColor:'#ff00ff',backgroundColor:'#ff00ff',pointStyle:'triangle',rotation:0,pointRadius:9,showLine:false,yAxisID:'y1'}},{{label:'RSI>80',data:r1.cSig80,borderColor:'#00dd00',backgroundColor:'#00dd00',pointStyle:'triangle',rotation:180,pointRadius:7,showLine:false,yAxisID:'y1'}},{{label:'RSI>85',data:r1.cSig85,borderColor:'#0070FF',backgroundColor:'#0070FF',pointStyle:'triangle',rotation:180,pointRadius:9,showLine:false,yAxisID:'y1'}}]}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{labels:{{color:'#ccc',filter:function(item){{const t=item.text;return t.includes('Index')||t.includes('MA200')||t.includes('RSI(10)');}}}}}}}},scales:{{x:{{ticks:{{color:'#666',maxTicksLimit:10}}}},y:{{type:'linear',display:true,position:'left',grid:{{color:'rgba(255,255,255,.05)'}},ticks:{{color:'#888'}}}},y1:{{type:'linear',display:true,position:'right',min:0,max:100,grid:{{display:false}},ticks:{{color:'#ffaa00'}}}}}}}}}});}}
initUI();
</script>
</body></html>"""
    return html_content


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 S&P 500 & Nasdaq 100 定投及增益策略")
    print(f"📊 标的: {ALL_ETFS}")
    print("=" * 60)

    try:
        dm = CSVDataManager()
        usd_hkd = get_usd_hkd_rate()
        print(f"💱 汇率: {usd_hkd:.4f}")

        print("\n📥 更新指数...")
        sp_index = update_ticker_data(dm, SP_INDEX_TICKER, earliest_date=INDEX_START_DATE)
        sp_index = calculate_indicators(sp_index)
        ndx_index = update_ticker_data(dm, NDX_INDEX_TICKER, earliest_date=INDEX_START_DATE)
        ndx_index = calculate_indicators(ndx_index)

        print("\n📥 更新ETF...")
        etf_data_map = {}
        for ticker in ALL_ETFS:
            if ticker in TICKER_MERGE_MAP:
                etf_data_map[ticker] = update_merged_ticker(dm, ticker, TICKER_MERGE_MAP[ticker])
            else:
                etf_data_map[ticker] = update_ticker_data(dm, ticker)

        json_data = prepare_json_data(sp_index, ndx_index, etf_data_map)
        html_str = generate_interactive_html(json_data, usd_hkd)
        with open(HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(html_str)
        print(f"\n✅ 完成! {HTML_PATH}")

        if not IS_CI:
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(HTML_PATH))
            input("按回车键退出...")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 出错: {e}")
        if not IS_CI:
            input("按回车键退出...")
