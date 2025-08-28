/* static/js/app.js */

/* ---------- helpers ---------- */
function getSelectedTickers() {
  const sel = document.getElementById("tickers");
  return Array.from(sel.selectedOptions).map((o) => o.value);
}

function ymd(d) {
  // normalize any input to YYYY-MM-DD for the backend
  if (!d) return "";
  const dt = new Date(d);
  return dt.toISOString().split("T")[0];
}

function setDefaultDates() {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - 365);
  document.getElementById("start").value = ymd(start);
  document.getElementById("end").value = ymd(end);
}

function paramsCommon() {
  return {
    start: ymd(document.getElementById("start").value),
    end: ymd(document.getElementById("end").value),
    interval: document.getElementById("interval").value,
    sma20: document.getElementById("sma20").checked ? "1" : "0",
    sma50: document.getElementById("sma50").checked ? "1" : "0",
    ema20: document.getElementById("ema20").checked ? "1" : "0",
    ema50: document.getElementById("ema50").checked ? "1" : "0",
    rsi: document.getElementById("rsi").checked ? "1" : "0",
    normalize: document.getElementById("normalize").checked ? "1" : "0"
  };
}

function showError(el, title, detail) {
  el.innerHTML =
    '<div class="alert alert-danger"><strong>' +
    title +
    "</strong>" +
    (detail ? '<div class="small mt-1">' + detail + "</div>" : "") +
    "</div>";
}

/* ---------- OVERVIEW (price + RSI + KPIs) ---------- */
async function loadOverview() {
  const root = document.getElementById("overview-root");
  root.innerHTML = '<div class="text-muted">Loading…</div>';

  const tickers = getSelectedTickers();
  if (tickers.length === 0) {
    showError(root, "No tickers selected");
    return;
  }

  const q = new URLSearchParams(Object.assign({ tickers: tickers.join(",") }, paramsCommon()));

  let res;
  try {
    res = await fetch("/api/data?" + q.toString());
  } catch (e) {
    showError(root, "Network error", String(e));
    return;
  }
  if (!res.ok) {
    const text = await res.text();
    showError(root, "API error (" + res.status + ")", text.slice(0, 250));
    return;
  }

  const data = await res.json();
  root.innerHTML = "";

  (data.tickers || []).forEach(function (t) {
    const card = document.createElement("div");
    card.className = "card shadow-sm mb-4";

    var kpiHtml = "";
    if (t.kpis) {
      kpiHtml =
        '<div><small>Close</small><div class="fs-5 fw-semibold">' +
        t.kpis.latest_close +
        "</div></div>" +
        '<div><small>Change %</small><div class="fs-5 fw-semibold">' +
        (t.kpis.pct_change != null ? t.kpis.pct_change : "-") +
        "%</div></div>" +
        '<div><small>52W High</small><div class="fs-5 fw-semibold">' +
        t.kpis.high_52w +
        "</div></div>" +
        '<div><small>52W Low</small><div class="fs-5 fw-semibold">' +
        t.kpis.low_52w +
        "</div></div>";
    } else {
      kpiHtml = '<span class="text-warning">No KPIs</span>';
    }

    var body =
      '<div class="card-body">' +
      '<div class="d-flex justify-content-between flex-wrap align-items-center">' +
      '<h5 class="card-title mb-3 mb-md-0">' +
      t.symbol +
      "</h5>" +
      '<div class="d-flex gap-3 flex-wrap kpi">' +
      kpiHtml +
      "</div></div>";

    if (t.error) {
      body += '<div class="alert alert-warning mt-3">' + t.error + "</div>";
    } else {
      body +=
        '<div id="price_' +
        t.symbol +
        '" class="mt-3"></div>' +
        (t.series && t.series.rsi
          ? '<div id="rsi_' + t.symbol + '" class="mt-3"></div>'
          : "") +
        '<div class="mt-3">' +
        '<a class="btn btn-outline-primary btn-sm" href="/download/csv?symbol=' +
        t.symbol +
        "&" +
        q.toString() +
        '">Download CSV</a>' +
        "</div>";
    }

    body += "</div>"; // card-body
    card.innerHTML = body;
    root.appendChild(card);

    if (t.error || !t.series) return;

    const s = t.series;
    const traces = [
      {
        x: s.index,
        open: s.open,
        high: s.high,
        low: s.low,
        close: s.close,
        type: "candlestick",
        name: t.symbol + " Price"
      }
    ];
    ["sma20", "sma50", "ema20", "ema50"].forEach(function (k) {
      if (s[k]) {
        traces.push({
          x: s.index,
          y: s[k],
          type: "scatter",
          mode: "lines",
          name: k.toUpperCase()
        });
      }
    });
    if (s.volume && s.volume.some(function (v) { return v !== null && v !== 0; })) {
      traces.push({
        x: s.index,
        y: s.volume,
        type: "bar",
        name: "Volume",
        opacity: 0.3,
        yaxis: "y2"
      });
    }

    Plotly.newPlot("price_" + t.symbol, traces, {
      height: 560,
      margin: { l: 10, r: 10, t: 40, b: 20 },
      xaxis: { rangeslider: { visible: false } },
      yaxis: {
        title: document.getElementById("normalize").checked
          ? "Index (Start=100)"
          : "Price"
      },
      yaxis2: { title: "Volume", overlaying: "y", side: "right", showgrid: false },
      legend: { orientation: "h" },
      title: t.symbol + " Price Chart"
    });

    if (s.rsi) {
      var firstIdx = s.index[0];
      var lastIdx = s.index[s.index.length - 1];
      Plotly.newPlot(
        "rsi_" + t.symbol,
        [
          { x: s.index, y: s.rsi, type: "scatter", mode: "lines", name: "RSI 14" },
          { x: [firstIdx, lastIdx], y: [70, 70], type: "scatter", mode: "lines", line: { dash: "dash" }, name: "70" },
          { x: [firstIdx, lastIdx], y: [30, 30], type: "scatter", mode: "lines", line: { dash: "dash" }, name: "30" }
        ],
        {
          height: 250,
          margin: { l: 10, r: 10, t: 40, b: 20 },
          yaxis: { title: "RSI" },
          showlegend: false,
          title: t.symbol + " RSI"
        }
      );
    }
  });
}

/* ---------- AUDIT ---------- */
async function loadAudit() {
  const root = document.getElementById("audit-root");
  root.innerHTML = '<div class="text-muted">Auditing…</div>';

  const tickers = getSelectedTickers();
  if (tickers.length === 0) {
    showError(root, "No tickers selected");
    return;
  }

  const common = paramsCommon();
  root.innerHTML = "";

  for (var i = 0; i < tickers.length; i++) {
    const symbol = tickers[i];
    const q = new URLSearchParams({
      symbol: symbol,
      start: common.start,
      end: common.end,
      interval: common.interval
    });

    let res;
    try {
      res = await fetch("/api/audit?" + q.toString());
    } catch (e) {
      const div = document.createElement("div");
      div.className = "alert alert-danger";
      div.textContent = "Audit network error for " + symbol + ": " + String(e);
      root.appendChild(div);
      continue;
    }

    if (!res.ok) {
      const text = await res.text();
      const div = document.createElement("div");
      div.className = "alert alert-danger";
      div.textContent = "Audit error for " + symbol + ": " + res.status + " " + text;
      root.appendChild(div);
      continue;
    }

    const data = await res.json();

    const card = document.createElement("div");
    card.className = "card shadow-sm mb-4";

    function badge(level, msg) {
      var variant = "info";
      if (level === "success") variant = "success";
      else if (level === "warning") variant = "warning";
      return '<span class="badge text-bg-' + variant + ' me-2 mb-2">' + msg + "</span>";
    }

    var alertsHtml = "";
    (data.alerts || []).forEach(function (a) {
      alertsHtml += badge(a.level, a.msg);
    });
    if (!alertsHtml) alertsHtml = '<span class="text-muted">No alerts.</span>';

    var gapsRows = "";
    (data.tables && data.tables.recent_gaps ? data.tables.recent_gaps : []).forEach(function (r) {
      gapsRows += "<tr><td>" + r.date + '</td><td class="text-end">' + r.gap_pct + "%</td></tr>";
    });
    if (!gapsRows) gapsRows = '<tr><td colspan="2" class="text-muted">None</td></tr>';

    var movesRows = "";
    (data.tables && data.tables.large_moves ? data.tables.large_moves : []).forEach(function (r) {
      movesRows += "<tr><td>" + r.date + '</td><td class="text-end">' + r.move_pct + "%</td></tr>";
    });
    if (!movesRows) movesRows = '<tr><td colspan="2" class="text-muted">None</td></tr>';

    card.innerHTML =
      '<div class="card-body">' +
      '<div class="d-flex justify-content-between flex-wrap align-items-center">' +
      '<h5 class="card-title mb-2">' +
      data.symbol +
      " · Audit</h5>" +
      '<div class="d-flex gap-3 flex-wrap kpi">' +
      '<div><small>Close</small><div class="fs-6 fw-semibold">' +
      (data.metrics.last_close != null ? data.metrics.last_close : "-") +
      '</div></div><div><small>Vol(20) ann.</small><div class="fs-6 fw-semibold">' +
      (data.metrics.vol_20_annual_pct != null ? data.metrics.vol_20_annual_pct : "-") +
      '%</div></div><div><small>Max DD</small><div class="fs-6 fw-semibold">' +
      (data.metrics.max_drawdown_pct != null ? data.metrics.max_drawdown_pct : "-") +
      '%</div></div><div><small>ATR(14)</small><div class="fs-6 fw-semibold">' +
      (data.metrics.atr14_pct != null ? data.metrics.atr14_pct : "-") +
      '%</div></div><div><small>52W High</small><div class="fs-6 fw-semibold">' +
      (data.metrics["52w_high"] != null ? data.metrics["52w_high"] : "-") +
      '</div></div><div><small>52W Low</small><div class="fs-6 fw-semibold">' +
      (data.metrics["52w_low"] != null ? data.metrics["52w_low"] : "-") +
      "</div></div></div></div>" +
      '<div class="mt-2">' +
      alertsHtml +
      "</div>" +
      '<div class="row mt-3 g-3">' +
      '<div class="col-md-6"><h6>Recent Gaps ≥ 2%</h6>' +
      '<div class="table-responsive"><table class="table table-sm table-dark table-striped align-middle mb-0">' +
      "<thead><tr><th>Date</th><th class=\"text-end\">Gap %</th></tr></thead>" +
      "<tbody>" +
      gapsRows +
      "</tbody></table></div></div>" +
      '<div class="col-md-6"><h6>Large Moves ≥ 3%</h6>' +
      '<div class="table-responsive"><table class="table table-sm table-dark table-striped align-middle mb-0">' +
      "<thead><tr><th>Date</th><th class=\"text-end\">Move %</th></tr></thead>" +
      "<tbody>" +
      movesRows +
      "</tbody></table></div></div>" +
      "</div></div>";

    root.appendChild(card);
  }
}

/* ---------- COMPARE (normalized + correlation) ---------- */
async function loadCompare() {
  const root = document.getElementById("compare-root");
  root.innerHTML = '<div class="text-muted">Loading comparison…</div>';

  const tickers = getSelectedTickers();
  if (tickers.length < 2) {
    showError(root, "Pick at least two tickers for comparison.");
    return;
  }

  const q = new URLSearchParams({
    tickers: tickers.join(","),
    start: ymd(document.getElementById("start").value),
    end: ymd(document.getElementById("end").value),
    interval: document.getElementById("interval").value
  });

  let res;
  try {
    res = await fetch("/api/compare?" + q.toString());
  } catch (e) {
    showError(root, "Network error", String(e));
    return;
  }
  if (!res.ok) {
    const t = await res.text();
    showError(root, "API error (" + res.status + ")", t);
    return;
  }

  const data = await res.json();
  root.innerHTML = "";

  const traces = [];
  Object.keys(data.series || {}).forEach(function (sym) {
    const obj = data.series[sym];
    traces.push({ x: obj.index, y: obj.values, type: "scatter", mode: "lines", name: sym });
  });

  const card1 = document.createElement("div");
  card1.className = "card shadow-sm mb-4";
  card1.innerHTML =
    '<div class="card-body"><h5 class="card-title">Normalized Performance (start=100)</h5><div id="norm_chart"></div></div>';
  root.appendChild(card1);

  Plotly.newPlot("norm_chart", traces, {
    height: 520,
    margin: { l: 10, r: 10, t: 40, b: 20 },
    legend: { orientation: "h" }
  });

  const labels = (data.corr && data.corr.labels) ? data.corr.labels : [];
  const z = (data.corr && data.corr.z) ? data.corr.z : [];

  const card2 = document.createElement("div");
  card2.className = "card shadow-sm mb-4";
  card2.innerHTML =
    '<div class="card-body"><h5 class="card-title">Return Correlations</h5><div id="corr_heat"></div></div>';
  root.appendChild(card2);

  Plotly.newPlot(
    "corr_heat",
    [{ z: z, x: labels, y: labels, type: "heatmap", hoverongaps: false }],
    { height: 520, margin: { l: 60, r: 10, t: 40, b: 60 } }
  );
}

/* ---------- boot ---------- */
async function loadAll() {
  await loadOverview();
  await loadAudit();
  await loadCompare();
}

document.addEventListener("DOMContentLoaded", function () {
  setDefaultDates();
  document.getElementById("loadBtn").addEventListener("click", function () {
    loadAll();
  });
  loadAll(); // auto-load once
});
