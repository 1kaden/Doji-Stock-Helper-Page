# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import time
from datetime import date, timedelta
from typing import List, Dict, Tuple

# Third-party components
try:
	from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:
	AgGrid = None
	GridOptionsBuilder = None
	GridUpdateMode = None

# Local imports (provided in signals.py)
from signals import tag_doji, compute_gap_pct, compute_nday_reversion


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="MatchaStocks – Gap Scanner", layout="wide")

# Session state defaults
if "gap_results" not in st.session_state:
	st.session_state["gap_results"] = None
if "gap_data_by_symbol" not in st.session_state:
	st.session_state["gap_data_by_symbol"] = {}
if "gap_selected" not in st.session_state:
	st.session_state["gap_selected"] = None
if "doji_results" not in st.session_state:
	st.session_state["doji_results"] = None
if "doji_data_by_symbol" not in st.session_state:
	st.session_state["doji_data_by_symbol"] = {}
if "doji_selected" not in st.session_state:
	st.session_state["doji_selected"] = None
if "last_gap_refresh" not in st.session_state:
	st.session_state["last_gap_refresh"] = 0.0
if "last_doji_refresh" not in st.session_state:
	st.session_state["last_doji_refresh"] = 0.0

st.markdown(
	"""
	### 🍵 MatchaStocks — Gap Up / Gap Down Scanner

	Find and filter daily gap ups/downs across your selected universe. Sort, search, and deep-dive into a ticker with a candlestick view.
	"""
)


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def fetch_sp500_tickers() -> List[str]:
	"""Fetch the latest S&P 500 tickers from Wikipedia."""
	url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
	tables = pd.read_html(url)
	df = tables[0]
	symbols = (
		df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
	)
	return symbols


def normalize_tickers(raw: str) -> List[str]:
	if not raw:
		return []
	items = [x.strip().upper() for x in raw.replace("\n", ",").split(",")]
	return [x for x in items if x]


@st.cache_data(show_spinner=False)
def download_history(
	tickers: Tuple[str, ...],
	start_dt: date,
	end_dt: date,
) -> Dict[str, pd.DataFrame]:
	"""Download OHLCV history for many tickers (daily interval). Returns dict per ticker."""
	if not tickers:
		return {}
	# yfinance can batch download which is faster
	data = yf.download(
		list(tickers),
		start=start_dt,
		end=end_dt + timedelta(days=1),  # ensure inclusivity for end day
		auto_adjust=False,
		group_by="ticker",
		threads=True,
		progress=False,
	)

	result: Dict[str, pd.DataFrame] = {}
	if isinstance(data.columns, pd.MultiIndex):
		for symbol in tickers:
			if symbol in data.columns.levels[0]:
				frame = data[symbol].dropna(how="all").copy()
				if not frame.empty:
					result[symbol] = frame
	else:
		# Single ticker shape
		frame = data.dropna(how="all").copy()
		if not frame.empty and len(tickers) == 1:
			result[list(tickers)[0]] = frame

	return result


def _store_selection(key: str, grid_response: dict):
	"""Persist selection from AgGrid into session state to survive reruns."""
	try:
		rows = grid_response.get("selected_rows", []) if isinstance(grid_response, dict) else []
		st.session_state[key] = rows[0] if rows else None
	except Exception:
		st.session_state[key] = None


def build_gap_table(
	data_by_symbol: Dict[str, pd.DataFrame],
	dir_filter: str,
	min_abs_gap_pct: float,
	include_doji: bool,
) -> pd.DataFrame:
	all_rows = []
	for symbol, df in data_by_symbol.items():
		if df.empty:
			continue
		# Enrich with optional signals
		if include_doji:
			df = tag_doji(df)
		# Compute gap percent vs previous close
		gaps = compute_gap_pct(df)
		df = df.assign(gap_pct=gaps).dropna(subset=["gap_pct"]).copy()

		prev_close = df["Close"].shift(1)
		for dt, row in df.iterrows():
			gap = float(row["gap_pct"])  # as fraction
			if dir_filter == "Up" and gap < min_abs_gap_pct:
				continue
			if dir_filter == "Down" and gap > -min_abs_gap_pct:
				continue
			if dir_filter == "Both" and abs(gap) < min_abs_gap_pct:
				continue

			all_rows.append(
				{
					"Symbol": symbol,
					"Date": dt.date().isoformat(),
					"Gap %": round(gap * 100.0, 2),
					"Direction": "Up" if gap > 0 else "Down",
					"Prev Close": float(prev_close.loc[dt]) if dt in prev_close.index else np.nan,
					"Open": float(row.get("Open", np.nan)),
					"High": float(row.get("High", np.nan)),
					"Low": float(row.get("Low", np.nan)),
					"Close": float(row.get("Close", np.nan)),
					"Volume": int(row.get("Volume", 0)) if not pd.isna(row.get("Volume", np.nan)) else 0,
					"Doji?": bool(row.get("is_doji", False)),
					"Doji Type": row.get("doji_type", None),
				}
			)

	return pd.DataFrame(all_rows)


def render_candlestick(
	symbol: str,
	center_date: str,
	window_days: int = 15,
	show_doji_markers: bool = True,
	n_day: int | None = None,
	tolerance_pct: float | None = None,
	show_volume: bool = False,
	show_ref_levels: bool = True,
):
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	center = pd.to_datetime(center_date).date()
	start = center - timedelta(days=window_days)
	end = center + timedelta(days=window_days)
	data = yf.download(symbol, start=start, end=end + timedelta(days=1), auto_adjust=False, progress=False)
	if data.empty:
		st.info("No chart data available.")
		return

	# Tag single-day doji types for marker overlays
	if show_doji_markers:
		data_tagged = tag_doji(data)
	else:
		data_tagged = data.copy()

	if show_volume:
		fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
		fig.add_trace(
			go.Candlestick(
				x=data.index,
				open=data["Open"],
				high=data["High"],
				low=data["Low"],
				close=data["Close"],
				name=symbol,
			),
			row=1,
			col=1,
		)
		fig.add_trace(
			go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#6baed6"),
			row=2,
			col=1,
		)
		fig.update_yaxes(title_text="Volume", row=2, col=1)
	else:
		fig = go.Figure(
			data=[
				go.Candlestick(
					x=data.index,
					open=data["Open"],
					high=data["High"],
					low=data["Low"],
					close=data["Close"],
					name=symbol,
				)
			]
		)

	# Overlay single-day doji markers
	if show_doji_markers and "is_doji" in data_tagged.columns:
		mask = data_tagged["is_doji"].fillna(False)
		if mask.any():
			# Compute gap % and range for hover
			gap_series = compute_gap_pct(data_tagged).fillna(0.0) * 100.0
			range_series = (data_tagged["High"] - data_tagged["Low"]).astype(float)
			fig.add_scatter(
				x=data_tagged.index[mask],
				y=data_tagged.loc[mask, "Close"],
				mode="markers",
				marker=dict(size=9, color="red", symbol="diamond", opacity=0.8),
				name="Doji",
				hovertext=[
					f"Type: {t}<br>Gap: {g:.2f}%<br>Range: {r:.2f}"
					for t, g, r in zip(
						data_tagged.loc[mask, "doji_type"].astype(str).fillna("").tolist(),
						gap_series.loc[mask].tolist(),
						range_series.loc[mask].tolist(),
					)
				],
				hoverinfo="x+y+text",
			)

	# Overlay N-day doji markers if requested
	if show_doji_markers and n_day and tolerance_pct is not None and n_day > 1:
		flags, delta = compute_nday_reversion(data_tagged, n_day, tolerance_pct / 100.0)
		mask_n = flags.fillna(False)
		if mask_n.any():
			fig.add_scatter(
				x=data_tagged.index[mask_n],
				y=data_tagged.loc[mask_n, "Close"],
				mode="markers",
				marker=dict(size=8, color="blue", symbol="circle", opacity=0.7),
				name=f"{n_day}-Day Doji",
				hovertext=[f"Δ vs start: {v:.2f}%" for v in (delta[mask_n] * 100.0).tolist()],
				hoverinfo="x+y+text",
			)

	# Add vertical highlight at the selected center date
	fig.add_vline(x=pd.to_datetime(center), line_width=1, line_dash="dash", line_color="orange")

	# Add reference levels (High/Low/Close) for the selected date as horizontal lines
	if show_ref_levels and pd.to_datetime(center) in pd.to_datetime(data.index).to_pydatetime():
		try:
			row = data.loc[pd.to_datetime(center)]
			c_close = float(row.get("Close", np.nan))
			c_high = float(row.get("High", np.nan))
			c_low = float(row.get("Low", np.nan))
			if not np.isnan(c_close):
				fig.add_hline(y=c_close, line_width=1, line_dash="dot", line_color="#ff7f0e", annotation_text=f"Close {c_close:.2f}", annotation_position="top right")
			if not np.isnan(c_high):
				fig.add_hline(y=c_high, line_width=1, line_dash="dash", line_color="#2ca02c", annotation_text=f"High {c_high:.2f}", annotation_position="top right")
			if not np.isnan(c_low):
				fig.add_hline(y=c_low, line_width=1, line_dash="dash", line_color="#d62728", annotation_text=f"Low {c_low:.2f}", annotation_position="bottom right")
		except Exception:
			pass

	fig.update_layout(
		height=520 if show_volume else 480,
		margin=dict(l=10, r=10, t=30, b=10),
		hovermode="x unified",
	)
	fig.update_xaxes(
		rangeslider=dict(visible=True),
		rangeselector=dict(
			buttons=[
				dict(count=5, label="5d", step="day", stepmode="backward"),
				dict(count=1, label="1m", step="month", stepmode="backward"),
				dict(count=3, label="3m", step="month", stepmode="backward"),
				dict(count=6, label="6m", step="month", stepmode="backward"),
				dict(count=1, label="YTD", step="year", stepmode="todate"),
				dict(count=1, label="1y", step="year", stepmode="backward"),
				dict(step="all"),
			]
		),
	)
	st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
	st.header("Scan Controls")
	mode = st.radio("Mode", ["Gap Scanner", "Doji Scanner"], index=0)
	universe_choice = st.selectbox("Universe", ["S&P 500 (Wikipedia)", "Custom Only"])

	custom_tickers_input = st.text_area(
		"Custom tickers (comma or newline separated)",
		"AAPL, MSFT, TSLA",
		height=80,
	)

	default_start = date.today() - timedelta(days=60)
	start_dt = st.date_input("Start Date", value=default_start)
	end_dt = st.date_input("End Date", value=date.today())

	dir_filter = st.selectbox("Gap Direction", ["Both", "Up", "Down"], index=0)
	min_gap_pct = st.slider("Min absolute gap %", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
	include_doji = st.checkbox("Compute Doji flag (adds Doji? column)", value=False)

	# Doji scanner specific controls
	if mode == "Doji Scanner":
		st.markdown("---")
		st.caption("Doji classification & multi-day settings")
		body_alpha = st.slider("Body vs range max (doji threshold)", 0.02, 0.3, 0.1, 0.01)
		wick_tol = st.slider("Wick equality tolerance (fraction of range)", 0.0, 0.3, 0.1, 0.01)
		long_wick_frac = st.slider("Long wick fraction of range", 0.3, 0.9, 0.6, 0.05)
		atr_floor = st.slider("Min range vs ATR (to avoid tiny bars)", 0.0, 2.0, 0.5, 0.1)
		ndoji_toggle = st.checkbox("Scan for multi-day dojis", value=False)
		ndoji_window = st.number_input("N-day window", min_value=1, max_value=30, value=5, step=1)
		ndoji_tolerance = st.slider("N-day tolerance % (|Close_t/Close_{t-N+1}-1|)", min_value=0.0, max_value=2.0, value=0.2, step=0.05)
	else:
		ndoji_toggle = False
		ndoji_window = 5
		ndoji_tolerance = 0.2

	scan_clicked = st.button("🔎 Run Scan", type="primary")


# -----------------------------
# Assemble universe
# -----------------------------
tickers: List[str] = []
try:
	if universe_choice.startswith("S&P 500"):
		tickers.extend(fetch_sp500_tickers())
except Exception as e:
	st.warning(f"Could not fetch S&P 500 list: {e}")

tickers.extend(normalize_tickers(custom_tickers_input))
tickers = sorted(list(dict.fromkeys([t for t in tickers if t])))

st.caption(f"Universe size: {len(tickers)} tickers")


# -----------------------------
# Run scan
# -----------------------------
auto_refresh = st.sidebar.checkbox("Live refresh (every 60s)", value=False)

if ((scan_clicked or (mode == "Gap Scanner" and auto_refresh and time.time() - st.session_state["last_gap_refresh"] > 60)) and mode == "Gap Scanner"):
	if not tickers:
		st.error("Please provide at least one ticker or choose a universe.")
		st.stop()

	with st.spinner("Downloading price history…"):
		data_by_symbol = download_history(tuple(tickers), start_dt, end_dt)
		st.session_state["gap_data_by_symbol"] = data_by_symbol
		st.session_state["last_gap_refresh"] = time.time()

	if not data_by_symbol:
		st.warning("No data downloaded. Try a different date range or tickers.")
		st.stop()

	with st.spinner("Computing gaps…"):
		results = build_gap_table(
			data_by_symbol=data_by_symbol,
			dir_filter=dir_filter,
			min_abs_gap_pct=min_gap_pct / 100.0,
			include_doji=include_doji,
		)
		st.session_state["gap_results"] = results

		# Enrich with N-day doji columns if requested
		if ndoji_toggle and not results.empty:
			flags_list = []
			delta_list = []
			for sym, frame in data_by_symbol.items():
				if frame.empty:
					continue
				flags, delta = compute_nday_reversion(frame, int(ndoji_window), ndoji_tolerance / 100.0)
				temp = pd.DataFrame({
					"Date": frame.index.date.astype(str),
					"Symbol": sym,
					"_ndoji_flag": flags.values,
					"_ndoji_delta": (delta * 100.0).values,
				})
				flags_list.append(temp)
			if flags_list:
				flags_df = pd.concat(flags_list, ignore_index=True)
				results = results.merge(flags_df, on=["Symbol", "Date"], how="left")
				results["N-Day Doji?"] = results["_ndoji_flag"].fillna(False).astype(bool)
				results["N-Day Δ% (vs start)"] = results["_ndoji_delta"].round(2)
				results.drop(columns=["_ndoji_flag", "_ndoji_delta"], inplace=True, errors="ignore")

# Render Gap Scanner table and visuals using cached state
if mode == "Gap Scanner":
	results = st.session_state.get("gap_results")
	st.subheader("Results")
	if results is None or results.empty:
		st.info("Run a scan to see results.")
	else:
		quick_search = st.text_input("Quick search (filters all columns)")
		if AgGrid and GridOptionsBuilder:
			gb = GridOptionsBuilder.from_dataframe(results)
			gb.configure_pagination(paginationAutoPageSize=True)
			gb.configure_side_bar()
			gb.configure_selection(selection_mode="single", use_checkbox=True)
			gb.configure_default_column(resizable=True, sortable=True, filter=True, floatingFilter=True)
			grid_options = gb.build()
			grid_response = AgGrid(
				results,
				gridOptions=grid_options,
				height=540,
				fit_columns_on_grid_load=True,
				update_mode=GridUpdateMode.SELECTION_CHANGED,
				allow_unsafe_jscode=True,
				quickFilterText=quick_search or "",
			)
			selected_rows = grid_response.get("selected_rows", []) if isinstance(grid_response, dict) else []
		else:
			st.dataframe(results, use_container_width=True)
			selected_rows = []

		if selected_rows:
			row = selected_rows[0]
			c1, c2 = st.columns([1, 2])
			with c1:
				st.markdown("**Selection**")
				cols_to_show = ["Symbol", "Date", "Gap %", "Direction", "Open", "Prev Close", "Close", "Volume", "Doji?"]
				st.write({k: row[k] for k in cols_to_show if k in row})
			with c2:
				show_vol = st.checkbox("Show volume", value=True)
				if st.button("Load Visuals", type="secondary"):
					render_candlestick(
						symbol=row["Symbol"],
						center_date=row["Date"],
						window_days=10,
						show_doji_markers=True,
						n_day=None,
						tolerance_pct=None,
						show_volume=show_vol,
					)


if ((scan_clicked or (mode == "Doji Scanner" and auto_refresh and time.time() - st.session_state["last_doji_refresh"] > 60)) and mode == "Doji Scanner"):
	if not tickers:
		st.error("Please provide at least one ticker or choose a universe.")
		st.stop()

	with st.spinner("Downloading price history…"):
		data_by_symbol = download_history(tuple(tickers), start_dt, end_dt)
		st.session_state["last_doji_refresh"] = time.time()

	if not data_by_symbol:
		st.warning("No data downloaded. Try a different date range or tickers.")
		st.stop()

	with st.spinner("Scanning dojis…"):
		rows = []
		for sym, df in data_by_symbol.items():
			if df.empty:
				continue
			# classify dojis using current thresholds
			df_doji = tag_doji(df, body_alpha=body_alpha, atr_floor=atr_floor, wick_tolerance=wick_tol, long_wick_fraction=long_wick_frac)
			if ndoji_toggle:
				flags, delta = compute_nday_reversion(df_doji, int(ndoji_window), ndoji_tolerance / 100.0)
				df_doji = df_doji.assign(is_nday_doji=flags, nday_delta_pct=(delta * 100.0))
			for dt, r in df_doji.iterrows():
				if not bool(r.get("is_doji", False)) and not bool(r.get("is_nday_doji", False)):
					continue
				rows.append({
					"Symbol": sym,
					"Date": dt.date().isoformat(),
					"Doji?": bool(r.get("is_doji", False)),
					"Doji Type": r.get("doji_type", None),
					"N-Day Doji?": bool(r.get("is_nday_doji", False)) if ndoji_toggle else False,
					"N-Day Δ% (vs start)": float(r.get("nday_delta_pct", np.nan)) if ndoji_toggle else np.nan,
					"Open": float(r.get("Open", np.nan)),
					"High": float(r.get("High", np.nan)),
					"Low": float(r.get("Low", np.nan)),
					"Close": float(r.get("Close", np.nan)),
					"Volume": int(r.get("Volume", 0)) if not pd.isna(r.get("Volume", np.nan)) else 0,
				})

	if not rows:
		st.info("No doji matches for the chosen parameters.")
		st.stop()

	doji_table = pd.DataFrame(rows)
	st.subheader("Doji Results")
	if AgGrid and GridOptionsBuilder:
		gb = GridOptionsBuilder.from_dataframe(doji_table)
		gb.configure_pagination(paginationAutoPageSize=True)
		gb.configure_side_bar()
		gb.configure_selection(selection_mode="single", use_checkbox=True)
		gb.configure_default_column(resizable=True, sortable=True, filter=True, floatingFilter=True)
		grid_options = gb.build()
		grid_response = AgGrid(
			doji_table,
			gridOptions=grid_options,
			height=540,
			fit_columns_on_grid_load=True,
			update_mode=GridUpdateMode.SELECTION_CHANGED,
			allow_unsafe_jscode=True,
		)
		selected_rows = grid_response.get("selected_rows", []) if isinstance(grid_response, dict) else []
	else:
		st.dataframe(doji_table, use_container_width=True)
		selected_rows = []

	if selected_rows:
		row = selected_rows[0]
		st.markdown("---")
		st.subheader(f"Doji Visuals — {row['Symbol']} on {row['Date']}")
		c1, c2 = st.columns([1, 2])
		with c1:
			st.write({k: row[k] for k in ["Doji?", "Doji Type", "N-Day Doji?", "N-Day Δ% (vs start)"] if k in row})
			show_vol = st.checkbox("Show volume", value=True, key="doji_show_vol")
			st.markdown("**Overlays**")
			ovl_doji = st.checkbox("Mark single-day dojis", value=True, key="ovl_doji")
			ovl_nday = st.checkbox("Mark N-day dojis", value=bool(ndoji_toggle), key="ovl_nday")
		with c2:
			if st.button("Load Visuals", type="secondary", key="doji_load_visuals"):
				render_candlestick(
					symbol=row["Symbol"],
					center_date=row["Date"],
					window_days=10,
					show_doji_markers=ovl_doji or ovl_nday,
					n_day=int(ndoji_window) if (ovl_nday and ndoji_toggle) else None,
					tolerance_pct=float(ndoji_tolerance) if ndoji_toggle else None,
					show_volume=show_vol,
					show_ref_levels=True,
				)


