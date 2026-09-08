"""Concrete task implementation for missing value imputation."""

from pathlib import Path

import pandas as pd

from masters.tasks.base import BaseTask, TaskEvaluation
from masters.tasks.metrics import compute_categorical_accuracy, compute_nrmse


class ImputationTask(BaseTask):
    """Missing value imputation task on tabular datasets."""

    def __init__(
        self,
        task_id: str,
        dataset_name: str,
        clean_df: pd.DataFrame,
        masked_df: pd.DataFrame,
        mask: pd.DataFrame,
        mechanism: str = "MCAR",
        rate: float = 0.15,
    ) -> None:
        self.task_id = task_id
        self.task_name = f"imputation_{dataset_name}_{mechanism}_{int(rate * 100)}"
        self.dataset_name = dataset_name
        self._clean_df = clean_df.copy()
        self._masked_df = masked_df.copy()
        self._mask = mask.copy()
        self.mechanism = mechanism
        self.rate = rate

    def get_input_dataframe(self) -> pd.DataFrame:
        """Return the masked input DataFrame."""
        return self._masked_df.copy()

    def generate_prompt(self) -> str:
        """Generate problem specification prompt for the LLM."""
        dtypes_summary = "\n".join(
            f"- `{col}`: {dtype} ({self._masked_df[col].isna().sum()} missing)"
            for col, dtype in self._masked_df.dtypes.items()
        )
        return (
            "You are a data scientist. Write a Python function "
            "`impute_missing(df: pd.DataFrame) -> pd.DataFrame` "
            "that handles missing values in the provided dataset.\n\n"
            f"Dataset Name: {self.dataset_name}\n"
            f"Dimensions: {self._masked_df.shape[0]} rows, "
            f"{self._masked_df.shape[1]} columns.\n"
            f"Columns and Missingness:\n{dtypes_summary}\n\n"
            "Requirements:\n"
            "1. Define exactly:\n"
            "   `def impute_missing(df: pd.DataFrame) -> pd.DataFrame:`\n"
            "2. Return a transformed DataFrame with zero missing values (NaNs).\n"
            "3. Preserve original column names, row order, and dimensions.\n"
            "4. Use appropriate statistical or ML strategies (e.g. median, KNN) "
            "fitted on available data.\n"
            "5. Return only valid Python code inside markdown code blocks."
        )

    def evaluate_output(self, output_df: pd.DataFrame) -> TaskEvaluation:
        """Evaluate syntactic validity, data invariants, and NRMSE quality."""
        if not isinstance(output_df, pd.DataFrame):
            return TaskEvaluation(
                syntax_success=True,
                data_valid=False,
                quality_score=None,
                details={"error": "Output is not a pandas DataFrame"},
            )

        if output_df.shape != self._clean_df.shape:
            err_msg = (
                f"Shape mismatch: expected {self._clean_df.shape}, "
                f"got {output_df.shape}"
            )
            return TaskEvaluation(
                syntax_success=True,
                data_valid=False,
                quality_score=None,
                details={"error": err_msg},
            )

        if list(output_df.columns) != list(self._clean_df.columns):
            return TaskEvaluation(
                syntax_success=True,
                data_valid=False,
                quality_score=None,
                details={"error": "Column names or order do not match"},
            )

        nan_count = int(output_df.isna().sum().sum())
        if nan_count > 0:
            return TaskEvaluation(
                syntax_success=True,
                data_valid=False,
                quality_score=None,
                details={"error": f"Incomplete imputation: {nan_count} NaNs remain"},
            )

        nrmse = compute_nrmse(self._clean_df, output_df, self._mask)
        cat_acc = compute_categorical_accuracy(self._clean_df, output_df, self._mask)

        return TaskEvaluation(
            syntax_success=True,
            data_valid=True,
            quality_score=nrmse,
            details={"nrmse": nrmse, "cat_accuracy": cat_acc},
        )

    def wrap_execution_code(
        self,
        function_code: str,
        input_csv: Path,
        output_csv: Path,
    ) -> str:
        """Wrap model function with IO loader and execution logic."""
        return (
            f"import pandas as pd\n"
            f"import numpy as np\n\n"
            f"{function_code}\n\n"
            f"if __name__ == '__main__':\n"
            f"    df_in = pd.read_csv(r'{input_csv}')\n"
            f"    df_out = impute_missing(df_in)\n"
            f"    df_out.to_csv(r'{output_csv}', index=False)\n"
        )
