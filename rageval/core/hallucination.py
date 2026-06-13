# rageval/core/hallucination.py

from enum import Enum
from dataclasses import dataclass


class HallucinationType(str, Enum):
    FACTUAL_ERROR = "factual_error"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    FABRICATED_DETAIL = "fabricated_detail"


@dataclass
class Hallucination:
    claim: str
    type: HallucinationType
    severity: float  # 0.0 to 1.0
    reason: str
