# rageval/core/retrieved_doc.py

from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    content: str
    source: str = "unknown"
    score: float = 1.0
    doc_id: str = ""
