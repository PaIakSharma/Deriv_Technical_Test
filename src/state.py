"""Pipeline stage FSM.

Every stage transition must go through ``Pipeline.advance``. The pipeline
refuses to skip ahead — by construction the report cannot be produced
before metrics exist, and metrics cannot be produced before ledgers exist.
"""
from __future__ import annotations

from enum import Enum
from typing import List


class Stage(Enum):
    INIT = 0
    STRATEGIES_LOADED = 1
    DATA_FETCHED_OR_SIMULATED = 2
    STRATEGIES_FORMALISED = 3
    SPECS_VALIDATED = 4
    BACKTESTS_EXECUTED = 5
    LEDGERS_WRITTEN = 6
    METRICS_COMPUTED = 7
    STRATEGIES_CRITIQUED = 8
    OPTIONAL_ROBUSTNESS_TESTS_COMPLETE = 9
    REPORT_GENERATED = 10
    VALIDATION_COMPLETE = 11
    RESULTS_FINALISED = 12


class PipelineStateError(RuntimeError):
    pass


class Pipeline:
    """A monotonic stage tracker that rejects out-of-order transitions."""

    def __init__(self) -> None:
        self.stage = Stage.INIT
        self.history: List[Stage] = [Stage.INIT]

    def advance(self, target: Stage) -> None:
        if target.value != self.stage.value + 1:
            raise PipelineStateError(
                f"Illegal transition {self.stage.name} -> {target.name}; "
                f"stages must be advanced one at a time in order."
            )
        self.stage = target
        self.history.append(target)

    def require(self, minimum: Stage) -> None:
        """Assert the pipeline has at least reached ``minimum``."""
        if self.stage.value < minimum.value:
            raise PipelineStateError(
                f"Operation requires stage >= {minimum.name}, "
                f"current stage is {self.stage.name}."
            )

    def __str__(self) -> str:
        return self.stage.name
