"""Utilities for computing ATR, doji flags, and gap percentages.

This file is encoded as UTF-8.
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def average_true_range(data: pd.DataFrame, period: int = 14) -> pd.Series:
	"""Compute a simple ATR (no Wilder smoothing) over daily bars."""
	high = data["High"].astype(float)
	low = data["Low"].astype(float)
	close = data["Close"].astype(float)
	prev_close = close.shift(1)
	true_range = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
	atr = true_range.rolling(period, min_periods=period).mean()
	return atr


def tag_doji(
	data: pd.DataFrame,
	body_alpha: float = 0.1,
	atr_floor: float = 0.5,
	wick_tolerance: float = 0.1,
	long_wick_fraction: float = 0.6,
) -> pd.DataFrame:
	"""
	Classify doji candles and tag rows.

	- is_doji: small body vs total range and non-trivial range vs ATR
	- doji_type: one of {"Dragonfly", "Gravestone", "Long-legged", "Doji"} or None

	Parameters are expressed as fractions of total range, except atr_floor (fraction of ATR).
	"""
	frame = data.copy()
	frame["ATR"] = average_true_range(frame, 14)
	open_px = frame["Open"].astype(float)
	high_px = frame["High"].astype(float)
	low_px = frame["Low"].astype(float)
	close_px = frame["Close"].astype(float)
	body = (close_px - open_px).abs()
	range_ = (high_px - low_px).clip(lower=1e-9)
	frame["is_doji"] = (body <= body_alpha * range_) & (range_ >= atr_floor * frame["ATR"])

	upper_wick = (high_px - np.maximum(open_px, close_px)).clip(lower=0.0)
	lower_wick = (np.minimum(open_px, close_px) - low_px).clip(lower=0.0)

	# Initialize type as None
	doji_type: pd.Series = pd.Series([None] * len(frame), index=frame.index, dtype=object)

	# Dragonfly: open≈close≈high, long lower wick
	dragonfly_mask = (
		frame["is_doji"]
		& ((high_px - np.maximum(open_px, close_px)).abs() <= wick_tolerance * range_)
		& (lower_wick >= long_wick_fraction * range_)
	)
	doji_type.loc[dragonfly_mask] = "Dragonfly"

	# Gravestone: open≈close≈low, long upper wick
	gravestone_mask = (
		frame["is_doji"]
		& ((np.minimum(open_px, close_px) - low_px).abs() <= wick_tolerance * range_)
		& (upper_wick >= long_wick_fraction * range_)
	)
	doji_type.loc[gravestone_mask] = "Gravestone"

	# Long-legged: both wicks are large
	long_legged_mask = (
		frame["is_doji"]
		& (doji_type.isna())
		& (upper_wick >= 0.4 * range_)
		& (lower_wick >= 0.4 * range_)
	)
	doji_type.loc[long_legged_mask] = "Long-legged"

	# Standard doji for remaining is_doji
	standard_mask = frame["is_doji"] & doji_type.isna()
	doji_type.loc[standard_mask] = "Doji"

	frame["doji_type"] = doji_type
	return frame


def compute_gap_pct(data: pd.DataFrame) -> pd.Series:
	"""Gap percentage as fraction: (Open - PrevClose) / PrevClose."""
	prev_close = data["Close"].astype(float).shift(1)
	open_px = data["Open"].astype(float)
	gap = (open_px - prev_close) / prev_close
	return gap


def compute_nday_reversion(
	data: pd.DataFrame,
	n_days: int,
	tolerance_frac: float,
):
	"""
	Compute N-day reversion where Close[t] ~= Close[t-(n_days-1)] within tolerance.
	Returns (is_nday_doji: Series[bool], delta_frac: Series[float]).
	"""
	close_px = data["Close"].astype(float)
	if n_days <= 1:
		return pd.Series([False] * len(close_px), index=data.index), pd.Series([np.nan] * len(close_px), index=data.index)
	start_close = close_px.shift(n_days - 1)
	delta_frac = (close_px / start_close) - 1.0
	is_nday = delta_frac.abs() <= tolerance_frac
	is_nday = is_nday & start_close.notna()
	return is_nday, delta_frac


