from datetime import datetime, timedelta, date
from io import BytesIO
from functools import lru_cache
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify, send_file, abort, Response

app = Flask(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA"]
VALID_INTERVALS = ["1d", "1wk", "1mo"]


# ---------- utils ----------
def _safe_date(s: str, fallback: date) -> date:
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return fallback


@lru_cache(maxsize=256)
def fetch_data_cached(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def add_indicators(
    df: pd.DataFrame, sma20: bool, sma50: bool, ema20: bool, ema50: bool, rsi: bool
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if sma20 and "Close" in out:
        out["SMA20"] = out["Close"].rolling(20, min_periods=1).mean()
    if sma50 and "Close" in out:
        out["SMA50"] = out["Close"].rolling(50, min_periods=1).mean()
    if ema20 and "Close" in out:
        out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    if ema50 and "Close" in out:
        out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    if rsi and "Close" in out:
        delta = out["Close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        out["RSI"] = 100 - (100 / (1 + rs))
        out["RSI"] = out["RSI"].fillna(method="bfill")
    return out


def _col_as_series(df: pd.DataFrame, name: str, length_fallback: Optional[int] = None) -> pd.Series:
    if name in df:
        s = df[name]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")
    index = df.index if length_fallback is None else range(length_fallback)
    return pd.Series(np.nan, index=index, dtype="float64")


def compute_kpis(df: pd.DataFrame):
    if df.empty or "Close" not in df:
        return None
    last_val = df["Close"].iloc[-1]
    first_val = df["Close"].iloc[0]
    latest = last_val.item() if hasattr(last_val, "item") else float(last_val)
    first = first_val.item() if hasattr(first_val, "item") else float(first_val)
    pct = ((latest - first) / first * 100) if first != 0 else np.nan
    window = min(252, len(df))
    hi_52 = df["Close"].tail(window).max()
    lo_52 = df["Close"].tail(window).min()
    return {
        "latest_close": round(float(latest), 2),
        "pct_change": round(float(pct), 2) if np.isfinite(pct) else None,
        "high_52w": round(float(hi_52), 2),
        "low_52w": round(float(lo_52), 2),
    }


def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    if plot_df.empty or "Close" not in plot_df:
        return plot_df
    base_val = plot_df["Close"].iloc[0]
    base = base_val.item() if hasattr(base_val, "item") else float(base_val)
    if base != 0:
        factor = 100.0 / base
        for col in ["Open", "High", "Low", "Close", "SMA20", "SMA50", "EMA20", "EMA50"]:
            if col in plot_df.columns:
                plot_df[col] = plot_df[col] * factor
    return plot_df


def df_to_series_payload(df: pd.DataFrame):
    idx = [ts.isoformat() for ts in pd.to_datetime(df.index).to_pydatetime()]
    n = len(idx)
    open_s = _col_as_series(df, "Open")
    high_s = _col_as_series(df, "High")
    low_s = _col_as_series(df, "Low")
    close_s = _col_as_series(df, "Close")
    vol_s = _col_as_series(df, "Volume")
    payload = {
        "index": idx,
        "open": open_s.round(4).where(open_s.notna(), None).tolist(),
        "high": high_s.round(4).where(high_s.notna(), None).tolist(),
        "low": low_s.round(4).where(low_s.notna(), None).tolist(),
        "close": close_s.round(4).where(close_s.notna(), None).tolist(),
        "volume": vol_s.fillna(0).round(4).tolist(),
    }
    for col in ["SMA20", "SMA50", "EMA20", "EMA50", "RSI"]:
        if col in df.columns:
            s = _col_as_series(df, col, n).round(4)
            payload[col.lower()] = s.where(s.notna(), None).tolist()
    return payload


# ---------- audit engine ----------
def compute_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict with metrics and audit findings for a single ticker df (daily)."""
    out: Dict[str, Any] = {"metrics": {}, "alerts": [], "tables": {}}
    if df.empty or "Close" not in df or len(df) < 30:
        out["alerts"].append({"level": "warning", "msg": "Insufficient data for full audit."})
        return out

    px = df["Close"].astype(float)
    ret = px.pct_change().dropna()

    # Core metrics
    vol_20 = float(ret.tail(20).std() * np.sqrt(252) * 100) if len(ret) >= 20 else np.nan  # annualized %
    dd = (px / px.cummax() - 1.0)
    max_drawdown = float(dd.min() * 100)

    # Bollinger (20, 2)
    ma20 = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_breach = (px > upper) | (px < lower)

    # MACD (12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_cross = ((macd.shift(1) < signal.shift(1)) & (macd > signal)) | ((macd.shift(1) > signal.shift(1)) & (macd < signal))

    # ATR(14)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / px) * 100

    # Gaps (>2%)
    if "Open" in df:
        prev_close = df["Close"].shift()
        gap_pct = ((df["Open"] - prev_close) / prev_close * 100).abs()
        gaps = gap_pct[gap_pct >= 2].dropna().tail(20)
    else:
        gaps = pd.Series(dtype=float)

    # 52w tests
    hi_52 = float(px.tail(min(252, len(px))).max())
    lo_52 = float(px.tail(min(252, len(px))).min())
    last = float(px.iloc[-1])

    # Large 1-day move (>3%)
    large_moves = ret[ret.abs() >= 0.03].tail(10)  # last 10 events

    # Populate metrics
    out["metrics"] = {
        "last_close": round(last, 2),
        "vol_20_annual_pct": round(vol_20, 2) if np.isfinite(vol_20) else None,
        "max_drawdown_pct": round(max_drawdown, 2),
        "atr14_pct": round(float(atr_pct.iloc[-1]), 2) if not atr_pct.empty else None,
        "52w_high": round(hi_52, 2),
        "52w_low": round(lo_52, 2),
    }

    # Alerts
    if bb_breach.iloc[-1]:
        out["alerts"].append({"level": "info", "msg": "Price outside Bollinger band (mean reversion watch)."})
    if macd_cross.iloc[-1]:
        out["alerts"].append({"level": "info", "msg": "MACD crossover detected."})
    if last >= hi_52:
        out["alerts"].append({"level": "success", "msg": "New 52-week high."})
    if last <= lo_52:
        out["alerts"].append({"level": "warning", "msg": "New 52-week low."})
    if not gaps.empty:
        out["alerts"].append({"level": "warning", "msg": f"{len(gaps)} gap(s) ≥ 2% in recent periods."})
    if not large_moves.empty:
        out["alerts"].append({"level": "warning", "msg": f"{len(large_moves)} daily move(s) ≥ 3% (recent)."})

    # Tables (as simple lists for JSON)
    out["tables"]["recent_gaps"] = [
        {"date": d.strftime("%Y-%m-%d"), "gap_pct": round(v, 2)} for d, v in gaps.items()
    ]
    out["tables"]["large_moves"] = [
        {"date": d.strftime("%Y-%m-%d"), "move_pct": round(v*100, 2)} for d, v in large_moves.items()
    ]
    return out


def compare_series(tickers: List[str], start: date, end: date, interval: str) -> Dict[str, Any]:
    """Return normalized close series and correlation matrix for tickers."""
    closes = {}
    for t in tickers:
        df = fetch_data_cached(t, start.isoformat(), end.isoformat(), interval)
        if not df.empty and "Close" in df:
            closes[t] = df["Close"].astype(float)
    if not closes:
        return {"series": {}, "corr": []}

    dfc = pd.DataFrame(closes).dropna(how="all")
    # Normalize to 100 at first available date per series
    norm = dfc.apply(lambda s: (s / s.dropna().iloc[0]) * 100.0)
    corr = dfc.pct_change().corr().round(3)

    series = {
        t: {
            "index": [ts.isoformat() for ts in norm.index.to_pydatetime()],
            "values": norm[t].round(4).where(norm[t].notna(), None).tolist(),
        }
        for t in norm.columns
    }
    corr_mat = {
        "labels": corr.columns.tolist(),
        "z": corr.values.tolist(),
    }
    return {"series": series, "corr": corr_mat}


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html", default_tickers=DEFAULT_TICKERS, intervals=VALID_INTERVALS)


@app.get("/favicon.ico")
def favicon_noop():
    return Response(status=204)


@app.get("/api/data")
def api_data():
    tickers_param = request.args.get("tickers", ",".join(DEFAULT_TICKERS))
    tickers = [t.strip() for t in tickers_param.split(",") if t.strip()]

    start = _safe_date(request.args.get("start", ""), date.today() - timedelta(days=365))
    end = _safe_date(request.args.get("end", ""), date.today())
    interval = request.args.get("interval", "1d")
    if interval not in VALID_INTERVALS:
        interval = "1d"

    sma20 = request.args.get("sma20", "1") == "1"
    sma50 = request.args.get("sma50", "0") == "1"
    ema20 = request.args.get("ema20", "0") == "1"
    ema50 = request.args.get("ema50", "0") == "1"
    rsi = request.args.get("rsi", "0") == "1"
    normalize = request.args.get("normalize", "0") == "1"

    result = {"tickers": []}
    for symbol in tickers:
        try:
            df = fetch_data_cached(symbol, start.isoformat(), end.isoformat(), interval)
            if df.empty:
                result["tickers"].append({"symbol": symbol, "error": "No data"})
                continue
            df = add_indicators(df, sma20, sma50, ema20, ema50, rsi)
            df_plot = normalize_price(df) if normalize else df
            kpis = compute_kpis(df_plot)
            payload = df_to_series_payload(df_plot)
            result["tickers"].append({"symbol": symbol, "kpis": kpis, "series": payload})
        except Exception as e:
            result["tickers"].append({"symbol": symbol, "error": f"{type(e).__name__}: {e}"})
    return jsonify(result)


@app.get("/api/audit")
def api_audit():
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        abort(400)
    start = _safe_date(request.args.get("start", ""), date.today() - timedelta(days=365))
    end = _safe_date(request.args.get("end", ""), date.today())
    interval = request.args.get("interval", "1d")
    if interval not in VALID_INTERVALS:
        interval = "1d"
    df = fetch_data_cached(symbol, start.isoformat(), end.isoformat(), interval)
    out = compute_audit(df)
    out["symbol"] = symbol
    return jsonify(out)


@app.get("/api/compare")
def api_compare():
    tickers_param = request.args.get("tickers", ",".join(DEFAULT_TICKERS))
    tickers = [t.strip() for t in tickers_param.split(",") if t.strip()]
    start = _safe_date(request.args.get("start", ""), date.today() - timedelta(days=365))
    end = _safe_date(request.args.get("end", ""), date.today())
    interval = request.args.get("interval", "1d")
    if interval not in VALID_INTERVALS:
        interval = "1d"
    out = compare_series(tickers, start, end, interval)
    return jsonify(out)


@app.get("/download/csv")
def download_csv():
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        abort(400)
    start = _safe_date(request.args.get("start", ""), date.today() - timedelta(days=365))
    end = _safe_date(request.args.get("end", ""), date.today())
    interval = request.args.get("interval", "1d")
    if interval not in VALID_INTERVALS:
        interval = "1d"
    sma20 = request.args.get("sma20", "1") == "1"
    sma50 = request.args.get("sma50", "0") == "1"
    ema20 = request.args.get("ema20", "0") == "1"
    ema50 = request.args.get("ema50", "0") == "1"
    rsi = request.args.get("rsi", "0") == "1"
    normalize = request.args.get("normalize", "0") == "1"

    df = fetch_data_cached(symbol, start.isoformat(), end.isoformat(), interval)
    if df.empty:
        abort(404)
    df = add_indicators(df, sma20, sma50, ema20, ema50, rsi)
    df = normalize_price(df) if normalize else df
    csv_bytes = df.to_csv().encode("utf-8")
    return send_file(
        BytesIO(csv_bytes), mimetype="text/csv", as_attachment=True,
        download_name=f"{symbol}_data.csv",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
