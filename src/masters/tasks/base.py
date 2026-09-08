"""Abstract base specifications and data structures for benchmark tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TaskEvaluation:
    """Outcome of evaluating a single task trial."""

    syntax_success: bool
    data_valid: bool
    quality_score: float | None
    details: dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract interface for all tabular data tasks."""

    task_id: str
    task_name: str

    @abstractmethod
    def generate_prompt(self) -> str:
        """Construct the prompt sent to the LLM."""

    @abstractmethod
    def get_input_dataframe(self) -> pd.DataFrame:
        """Return the input DataFrame provided to the model's generated code."""

    @abstractmethod
    def evaluate_output(self, output_df: pd.DataFrame) -> TaskEvaluation:
        """Validate DataFrame schema integrity and compute quality score."""
