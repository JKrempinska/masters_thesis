"""Evaluation metrics for measuring data transformation and imputation quality."""

import numpy as np
import pandas as pd


def compute_nrmse(
    ground_truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
) -> float:
    """Compute Normalized Root Mean Squared Error (NRMSE) on masked entries.

    NRMSE is normalized by the standard deviation of true masked values:
        NRMSE = RMSE(X_masked, X_hat_masked) / std(X_masked)

    Args:
        ground_truth: True, complete DataFrame.
        imputed: DataFrame produced after imputation.
        mask: Boolean DataFrame indicating where missing values were injected (True).

    Returns:
        Float NRMSE value. Lower indicates superior reconstruction.
    """
    numeric_cols = [
        col
        for col in ground_truth.columns
        if pd.api.types.is_numeric_dtype(ground_truth[col]) and mask[col].any()
    ]
    if not numeric_cols:
        return 0.0

    sq_errors: list[float] = []
    true_vals: list[float] = []

    for col in numeric_cols:
        col_mask = mask[col]
        true_col = ground_truth.loc[col_mask, col].to_numpy(dtype=float)
        try:
            imputed_col = imputed.loc[col_mask, col].to_numpy(dtype=float)
        except (ValueError, TypeError):
            return float("inf")

        sq_errors.extend((true_col - imputed_col) ** 2)
        true_vals.extend(true_col)

    if not sq_errors:
        return 0.0

    rmse = float(np.sqrt(np.mean(sq_errors)))
    std_val = float(np.std(true_vals))

    if std_val < 1e-12:
        return rmse
    return rmse / std_val


def compute_categorical_accuracy(
    ground_truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
) -> float:
    """Compute exact-match accuracy for masked categorical entries.

    Args:
        ground_truth: True, complete DataFrame.
        imputed: Imputed DataFrame.
        mask: Boolean DataFrame indicating masked cells (True).

    Returns:
        Fraction of correctly reconstructed categorical cells in [0.0, 1.0].
    """
    cat_cols = [
        col
        for col in ground_truth.columns
        if not pd.api.types.is_numeric_dtype(ground_truth[col]) and mask[col].any()
    ]
    if not cat_cols:
        return 1.0

    total_cells = 0
    correct_cells = 0

    for col in cat_cols:
        col_mask = mask[col]
        true_vals = ground_truth.loc[col_mask, col].astype(str)
        imputed_vals = imputed.loc[col_mask, col].astype(str)

        total_cells += int(col_mask.sum())
        correct_cells += int((true_vals == imputed_vals).sum())

    if total_cells == 0:
        return 1.0
    return correct_cells / total_cells
