# rageval/judges/__init__.py

from rageval.judges.base import BaseJudge
from rageval.judges.heuristic import HeuristicJudge

# This tells Python exactly what is available to import from this folder
__all__ = ["BaseJudge", "HeuristicJudge"]
