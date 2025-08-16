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


def tag_doji(data: pd.DataFrame, alpha: float = 0.1, atr_floor: float = 0.5) -> pd.DataFrame:
	"""
	Mark rows where candle body is small relative to range (doji-like) and the range is non-trivial
	relative to ATR. Returns a copied DataFrame with columns ["ATR", "is_doji"].
	"""
	frame = data.copy()
	frame["ATR"] = average_true_range(frame, 14)
	open_px = frame["Open"].astype(float)
	high_px = frame["High"].astype(float)
	low_px = frame["Low"].astype(float)
	close_px = frame["Close"].astype(float)
	body = (close_px - open_px).abs()
	range_ = (high_px - low_px).clip(lower=1e-9)
	frame["is_doji"] = (body <= alpha * range_) & (range_ >= atr_floor * frame["ATR"])
	return frame


def compute_gap_pct(data: pd.DataFrame) -> pd.Series:
	"""Gap percentage as fraction: (Open - PrevClose) / PrevClose."""
	prev_close = data["Close"].astype(float).shift(1)
	open_px = data["Open"].astype(float)
	gap = (open_px - prev_close) / prev_close
	return gap



