from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import yfinance as yf
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
PHI = 1.6180339887

# ── Core functions ────────────────────────────────────────────

def load(ticker):
    # آخر 10 سنوات
    df = yf.download(ticker, period='10y', auto_adjust=True, progress=False)
    if df.empty:
        return None
    df = df[['Close']].reset_index()
    df.columns = ['Date', 'Close']
    df['Close'] = df['Close'].squeeze()
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)

def stretch(arr, n):
    if len(arr) < 2 or n < 2:
        return arr
    return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(arr)), arr)

def pearson_r(a, b):
    n = min(len(a), len(b))
    if n < 10:
        return None
    r, _ = pearsonr(a[:n], b[:n])
    return round(r * 100, 1)

def digit_sum(n):
    return sum(int(d) for d in str(n))

def phi_label(p):
    return ['1x','φ¹ 1.618x','φ² 2.618x','φ³ 4.236x','φ⁴ 6.854x','φ⁵ 11.09x'][p]

def score_emoji(r):
    if r is None: return '⬜'
    if r > 70:    return '✅'
    if r > 40:    return '🟡'
    if r > 0:     return '🟠'
    return               '❌'

def score_color(r):
    if r is None: return '#888'
    if r > 70:    return '#4caf50'
    if r > 40:    return '#ff9800'
    if r > 0:     return '#ff5722'
    return               '#f44336'

def find_best(df):
    today = pd.Timestamp.today()
    candidates = []
    for _, row in df.iterrows():
        d = row['Date']
        if d > today:
            continue
        sy  = digit_sum(d.year)
        smd = d.month + d.day
        if sy == 11 or smd == 11:
            candidates.append(d)

    future_results  = []  # بصمات لها إسقاط مستقبلي
    history_results = []  # بصمات تاريخية للمقارنة فقط

    for sd in candidates:
        for bm in [3, 5, 8, 10, 13]:
            try:
                p0    = float(df[df['Date'] == sd]['Close'].iloc[0])
                end_b = sd + pd.DateOffset(months=bm)
                basma = df[(df['Date'] >= sd) & (df['Date'] < end_b)]
                real  = df[df['Date'] >= end_b].reset_index(drop=True)
                if len(basma) < 10 or len(real) < 10:
                    continue
                p0r = float(real['Close'].iloc[0])
                bw  = (basma.set_index('Date')['Close'].resample('W').last().dropna() / p0 - 1) * 100
                rw  = (real.set_index('Date')['Close'].resample('W').last().dropna() / p0r - 1) * 100

                for pp in range(6):
                    ratio  = PHI ** pp
                    n_proj = int(len(bw) * ratio)
                    if n_proj < 5:
                        continue

                    # عدد الأسابيع المعروفة (حدثت فعلاً)
                    n_known = min(n_proj, len(rw))
                    if n_known < 10:
                        continue

                    proj = stretch(bw.values, n_proj)
                    r    = pearson_r(proj, rw.values[:n_known])
                    if r and r > 80:
                        has_future = n_proj > len(rw)
                        entry = {
                            'date':       sd.strftime('%Y-%m-%d'),
                            'months':     bm,
                            'phi':        pp,
                            'r':          r,
                            'has_future': has_future,
                        }
                        if has_future:
                            future_results.append(entry)
                        else:
                            history_results.append(entry)
            except:
                continue

    future_results.sort(key=lambda x: x['r'], reverse=True)
    history_results.sort(key=lambda x: x['r'], reverse=True)

    # الأولوية للبصمات المستقبلية — وإلا أفضل تاريخية
    combined = future_results[:5] if future_results else history_results[:5]
    return combined

def project(df, best):
    sd    = pd.Timestamp(best['date'])
    bm    = best['months']
    pp    = best['phi']
    ratio = PHI ** pp
    end_b = sd + pd.DateOffset(months=bm)
    basma = df[(df['Date'] >= sd) & (df['Date'] < end_b)]
    real  = df[df['Date'] >= end_b].reset_index(drop=True)
    if len(basma) < 5:
        return None
    p0  = float(df[df['Date'] >= sd].iloc[0]['Close'])
    p0r = float(real['Close'].iloc[0]) if len(real) > 0 else p0
    bw  = (basma.set_index('Date')['Close'].resample('W').last().dropna() / p0 - 1) * 100
    n_proj = int(len(bw) * ratio)
    proj   = stretch(bw.values, n_proj)
    known  = min(len(real), n_proj) if len(real) > 0 else 0
    future = proj[known:]
    p_now  = float(df['Close'].iloc[-1])
    last_date = df['Date'].iloc[-1]
    milestones = []
    for i, val in enumerate(future):
        if i % 13 == 0 or i == len(future) - 1:
            weeks_ahead = i + 1
            fut_date  = last_date + pd.Timedelta(weeks=weeks_ahead)
            fut_pct   = val - (proj[known - 1] if known > 0 else 0)
            fut_price = p_now * (1 + fut_pct / 100)
            milestones.append((fut_date, fut_price, fut_pct))
    return {
        'p_now': p_now,
        'weeks_left': len(future),
        'milestones': milestones,
    }

# ── HTML Template ─────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BASMA PHI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif; padding: 20px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  h1 { color: #f7931a; font-size: 22px; margin-bottom: 4px; }
  h2 { color: #58a6ff; font-size: 16px; margin-bottom: 12px; }
  input[type=text] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #30363d; background: #0d1117; color: #e6edf3; font-size: 16px; margin-bottom: 10px; }
  button { width: 100%; padding: 12px; border-radius: 8px; border: none; background: #f7931a; color: #000; font-size: 16px; font-weight: bold; cursor: pointer; }
  button:hover { background: #e8851a; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { color: #8b949e; font-weight: normal; padding: 6px 8px; border-bottom: 1px solid #30363d; text-align: right; }
  td { padding: 8px; border-bottom: 1px solid #21262d; }
  .big { font-size: 28px; font-weight: bold; color: #f7931a; }
  .label { color: #8b949e; font-size: 12px; }
  .up   { color: #4caf50; }
  .down { color: #f44336; }
  .hint { color: #8b949e; font-size: 13px; margin-top: 8px; }
  .loading { text-align: center; color: #8b949e; padding: 20px; }
  .error { color: #f44336; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 12px; }
</style>
</head>
<body>

<div class="card">
  <h1>🧬 BASMA PHI</h1>
  <p class="label">نظرية البصمة والنسبة الذهبية</p>
</div>

<div class="card">
  <form method="POST">
    <input type="text" name="ticker" placeholder="رمز السهم: AAPL / BTC-USD / 2222.SR"
           value="{{ ticker or '' }}" autocomplete="off" autocapitalize="characters">
    <button type="submit" id="btn" onclick="showLoad()">🔍 تحليل</button>
  </form>
  <div id="loader" style="display:none;text-align:center;padding:20px;color:#f7931a">
    ⏳ جاري التحليل... (30-60 ثانية)
  </div>
  <script>
    function showLoad() {
      setTimeout(function(){
        document.getElementById('loader').style.display='block';
        document.getElementById('btn').disabled=true;
      }, 100);
    }
  </script>
  <p class="hint">أمثلة: MSFT · AAPL · BTC-USD · 2222.SR · 1120.SR · ETH-USD</p>
</div>

{% if error %}
<div class="card">
  <p class="error">❌ {{ error }}</p>
</div>
{% endif %}

{% if loading %}
<div class="card loading">
  <p>⏳ جاري التحليل... قد يستغرق دقيقة</p>
</div>
{% endif %}

{% if result %}

<div class="card">
  <h2>🧬 البصمة الجينية — {{ result.ticker }}</h2>
  <table>
    <tr><th>تاريخ البصمة</th><td>{{ result.date }}</td></tr>
    <tr><th>الطول</th><td>{{ result.months }} شهر</td></tr>
    <tr><th>معامل φ</th><td><b style="color:#f7931a">{{ result.phi_name }}</b></td></tr>
    <tr><th>التشابه</th>
        <td><b style="color:{{ result.r_color }}">{{ result.r }}% {{ result.r_emoji }}</b></td></tr>
    <tr><th>السعر الحالي</th>
        <td><span class="big">{{ result.p_now }}</span></td></tr>
  </table>
</div>

{% if result.milestones %}
<div class="card">
  <h2>📈 الإسقاط المستقبلي ({{ result.weeks_left }} أسبوع)</h2>
  <table>
    <tr>
      <th>التاريخ</th>
      <th>السعر المتوقع</th>
      <th>التغيير</th>
    </tr>
    {% for d, px, pct in result.milestones %}
    <tr>
      <td>{{ d.strftime('%Y-%m-%d') }}</td>
      <td><b>{{ "{:,.0f}".format(px) }}</b></td>
      <td class="{{ 'up' if pct >= 0 else 'down' }}">
        {{ '▲' if pct >= 0 else '▼' }} {{ "{:.1f}".format(pct|abs) }}%
      </td>
    </tr>
    {% endfor %}
  </table>
</div>
{% else %}
<div class="card">
  <p style="color:#f44336">⚠️ البيانات غير دقيقة — لا يوجد إسقاط مستقبلي بتشابه كافٍ لهذا السهم</p>
</div>
{% endif %}

{% if result.all_results %}
<div class="card">
  <h2>📊 أفضل النتائج</h2>
  <table>
    <tr><th>التاريخ</th><th>بصمة</th><th>φ</th><th>تشابه</th></tr>
    {% for x in result.all_results %}
    <tr>
      <td>{{ x.date }}</td>
      <td>{{ x.months }}م</td>
      <td>{{ x.phi_name }}</td>
      <td style="color:{{ x.r_color }}">{{ x.r }}% {{ x.r_emoji }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% endif %}

</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(HTML)

    ticker = request.form.get('ticker', '').strip().upper()
    if not ticker:
        return render_template_string(HTML, error="أدخل رمز السهم")

    df = load(ticker)
    if df is None:
        return render_template_string(HTML, ticker=ticker,
                                      error=f"لا توجد بيانات لـ {ticker}")

    results = find_best(df)
    if not results:
        return render_template_string(HTML, ticker=ticker,
                                      error="⚠️ لم توجد بصمة بتشابه كافٍ — البيانات غير دقيقة لهذا السهم")

    best = results[0]
    fut  = project(df, best)

    p_now = float(df['Close'].iloc[-1])
    if p_now >= 1000:
        p_now_str = f"{p_now:,.0f}"
    elif p_now >= 1:
        p_now_str = f"{p_now:,.2f}"
    else:
        p_now_str = f"{p_now:.4f}"

    all_res = []
    for x in results:
        all_res.append({
            'date':     x['date'],
            'months':   x['months'],
            'phi_name': phi_label(x['phi']),
            'r':        x['r'],
            'r_emoji':  score_emoji(x['r']),
            'r_color':  score_color(x['r']),
        })

    result = {
        'ticker':    ticker,
        'date':      best['date'],
        'months':    best['months'],
        'phi_name':  phi_label(best['phi']),
        'r':         best['r'],
        'r_emoji':   score_emoji(best['r']),
        'r_color':   score_color(best['r']),
        'p_now':     p_now_str,
        'milestones': fut['milestones'] if fut and fut['weeks_left'] > 0 else [],
        'weeks_left': fut['weeks_left'] if fut else 0,
        'all_results': all_res,
    }

    return render_template_string(HTML, ticker=ticker, result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
