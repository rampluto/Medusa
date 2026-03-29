"""MEDUSA (Medallion-Engineered Deterministic Unified Storage Agent) environment.

Full Bronze→Silver integration controller with:
- Multi-source join orchestration (inner / left / anti)
- Schema drift handling (EVOLVE_SCHEMA)
- Key preparation and deduplication
- SCD-1 and SCD-2 merge logic
- Per-step RL reward engine
- Deterministic post-commit grader
"""

from .client import medusa_env
from .grader import Grader, GraderResult
from .medusa_env import MedusaEnv
from .models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
from .rewards import RewardEngine
from .scenarios import Scenario, ScenarioGenerator
from .tasks import TASKS, Task, TaskResult, score_episode

__all__ = [
    "medusa_env",
    "MedusaEnv",
    "MedusaAction",
    "MedusaActionType",
    "MedusaObservation",
    "MedusaState",
    "Scenario",
    "ScenarioGenerator",
    "RewardEngine",
    "Grader",
    "GraderResult",
    "TASKS",
    "Task",
    "TaskResult",
    "score_episode",
]
