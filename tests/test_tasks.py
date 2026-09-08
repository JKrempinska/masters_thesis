"""Unit tests for task masking, metrics, and imputation task definitions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from masters.tasks import (
    ImputationTask,
    compute_categorical_accuracy,
    compute_nrmse,
    inject_mar,
    inject_mcar,
    inject_mnar,
)


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    """Fixture providing a deterministic numeric DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(10.0, 2.0, size=200),
            "feature_b": rng.uniform(0.0, 100.0, size=200),
            "category": np.where(rng.random(size=200) > 0.5, "cat", "dog"),
        }
    )


def test_inject_mcar(sample_numeric_df: pd.DataFrame) -> None:
    rate = 0.20
    masked, mask = inject_mcar(sample_numeric_df, rate=rate, seed=123)

    assert not sample_numeric_df["feature_a"].isna().any()
    assert masked["feature_a"].isna().any()
    assert mask.shape == sample_numeric_df.shape

    actual_rate = mask["feature_a"].mean()
    assert abs(actual_rate - rate) < 0.08


def test_inject_mcar_deterministic(sample_numeric_df: pd.DataFrame) -> None:
    m1, mask1 = inject_mcar(sample_numeric_df, rate=0.25, seed=99)
    m2, mask2 = inject_mcar(sample_numeric_df, rate=0.25, seed=99)
    assert mask1.equals(mask2)
    assert m1.equals(m2)


def test_inject_mcar_invalid_rate(sample_numeric_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        inject_mcar(sample_numeric_df, rate=1.5)
    with pytest.raises(ValueError):
        inject_mcar(sample_numeric_df, rate=-0.1)


def test_inject_mar(sample_numeric_df: pd.DataFrame) -> None:
    masked, mask = inject_mar(
        sample_numeric_df,
        target_col="feature_a",
        cond_col="feature_b",
        rate=0.30,
        seed=42,
    )
    assert masked["feature_a"].isna().any()
    assert not masked["feature_b"].isna().any()
    assert mask["feature_a"].any()


def test_inject_mnar(sample_numeric_df: pd.DataFrame) -> None:
    masked, mask = inject_mnar(
        sample_numeric_df,
        target_col="feature_a",
        rate=0.20,
        seed=42,
    )
    assert masked["feature_a"].isna().any()
    assert mask["feature_a"].any()


def test_compute_nrmse_perfect_imputation(sample_numeric_df: pd.DataFrame) -> None:
    masked, mask = inject_mcar(sample_numeric_df, rate=0.20, seed=42)
    score = compute_nrmse(sample_numeric_df, sample_numeric_df, mask)
    assert score == 0.0


def test_compute_nrmse_with_error(sample_numeric_df: pd.DataFrame) -> None:
    masked, mask = inject_mcar(sample_numeric_df, rate=0.20, seed=42)
    imputed = sample_numeric_df.copy()
    imputed.loc[mask["feature_a"], "feature_a"] += 5.0

    score = compute_nrmse(sample_numeric_df, imputed, mask)
    assert score > 0.0


def test_compute_categorical_accuracy() -> None:
    df_true = pd.DataFrame({"label": ["A", "B", "A", "B"]})
    df_pred = pd.DataFrame({"label": ["A", "A", "A", "B"]})
    mask = pd.DataFrame({"label": [True, True, True, True]})

    acc = compute_categorical_accuracy(df_true, df_pred, mask)
    assert acc == 0.75


def test_imputation_task_evaluation_flow(sample_numeric_df: pd.DataFrame) -> None:
    masked, mask = inject_mcar(sample_numeric_df, rate=0.15, seed=42)
    task = ImputationTask(
        task_id="task_001",
        dataset_name="synthetic",
        clean_df=sample_numeric_df,
        masked_df=masked,
        mask=mask,
        mechanism="MCAR",
        rate=0.15,
    )

    prompt = task.generate_prompt()
    assert "impute_missing" in prompt
    assert "synthetic" in prompt

    # Test perfect evaluation
    eval_clean = task.evaluate_output(sample_numeric_df)
    assert eval_clean.syntax_success is True
    assert eval_clean.data_valid is True
    assert eval_clean.quality_score == 0.0

    # Test remaining NaNs failure
    eval_incomplete = task.evaluate_output(masked)
    assert eval_incomplete.data_valid is False
    assert "NaNs remain" in str(eval_incomplete.details.get("error"))

    # Test shape mismatch failure
    eval_shape = task.evaluate_output(sample_numeric_df.iloc[:10])
    assert eval_shape.data_valid is False

    # Test wrapped execution script
    code = task.wrap_execution_code(
        "def impute_missing(df): return df.fillna(0)",
        Path("in.csv"),
        Path("out.csv"),
    )
    assert "impute_missing(df_in)" in code
