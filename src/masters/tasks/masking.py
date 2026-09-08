"""Synthetic missingness injection mechanisms (MCAR, MAR, MNAR) for tabular data."""

import numpy as np
import pandas as pd


def inject_mcar(
    df: pd.DataFrame,
    rate: float,
    columns: list[str] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject Missing Completely at Random (MCAR) values into a DataFrame.

    Each entry in the targeted columns has an independent probability `rate`
    of being set to NaN.

    Args:
        df: Input DataFrame. Will not be modified in-place.
        rate: Missingness fraction in (0.0, 1.0).
        columns: Specific column names to mask. If None, masks all columns.
        seed: Random seed for deterministic masking.

    Returns:
        Tuple of (masked_df, mask_bool_df), where mask_bool_df is True at
        positions where values were replaced with NaN.
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"Rate must be strictly between 0.0 and 1.0, got {rate}")

    rng = np.random.default_rng(seed)
    target_cols = list(columns) if columns is not None else list(df.columns)
    masked_df = df.copy()
    mask_df = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in target_cols:
        col_mask = rng.random(len(df)) < rate
        masked_df.loc[col_mask, col] = np.nan
        mask_df.loc[col_mask, col] = True

    return masked_df, mask_df


def inject_mar(
    df: pd.DataFrame,
    target_col: str,
    cond_col: str,
    rate: float,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject Missing at Random (MAR) values into `target_col` based on `cond_col`.

    Probability of missingness in `target_col` is higher for rows where
    `cond_col` exceeds its median value.

    Args:
        df: Input DataFrame.
        target_col: Column where missingness is injected.
        cond_col: Conditioning column determining missingness probability.
        rate: Overall target missingness rate in (0.0, 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (masked_df, mask_bool_df).
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"Rate must be strictly between 0.0 and 1.0, got {rate}")

    rng = np.random.default_rng(seed)
    masked_df = df.copy()
    mask_df = pd.DataFrame(False, index=df.index, columns=df.columns)

    median_val = df[cond_col].median()
    is_high = df[cond_col] >= median_val

    # Upper median rows get high probability; lower median rows get low probability
    high_prob = min(rate * 1.6, 0.95)
    low_prob = max(
        (rate * len(df) - high_prob * is_high.sum()) / (~is_high).sum(),
        0.01,
    )

    probs = np.where(is_high, high_prob, low_prob)
    col_mask = rng.random(len(df)) < probs

    masked_df.loc[col_mask, target_col] = np.nan
    mask_df.loc[col_mask, target_col] = True

    return masked_df, mask_df


def inject_mnar(
    df: pd.DataFrame,
    target_col: str,
    rate: float,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject Missing Not at Random (MNAR) values via self-censoring in `target_col`.

    Values in the upper percentile of `target_col` have a substantially higher
    probability of being missing.

    Args:
        df: Input DataFrame.
        target_col: Target column to self-censor.
        rate: Overall target missingness rate in (0.0, 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (masked_df, mask_bool_df).
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"Rate must be strictly between 0.0 and 1.0, got {rate}")

    rng = np.random.default_rng(seed)
    masked_df = df.copy()
    mask_df = pd.DataFrame(False, index=df.index, columns=df.columns)

    cutoff = df[target_col].quantile(1.0 - rate)
    is_upper = df[target_col] >= cutoff

    # Heavy bias towards values above cutoff
    probs = np.where(is_upper, 0.85, 0.05)
    col_mask = rng.random(len(df)) < probs

    masked_df.loc[col_mask, target_col] = np.nan
    mask_df.loc[col_mask, target_col] = True

    return masked_df, mask_df
