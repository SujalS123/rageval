from dataclasses import dataclass, field
from typing import Optional, Union
from rageval.core.retrieved_doc import RetrievedDoc


@dataclass
class RAGSample:
    """
    A single RAG interaction to be evaluated.

    retrieved_docs accepts either list[str] or list[RetrievedDoc].
    All metrics use sample.retrieved_texts internally — plain strings
    regardless of which type was passed. Fully backward compatible.
    """
    query: str
    retrieved_docs: list
    answer: str
    ground_truth: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError(f"Query must be a non-empty string. Got: {self.query}")
        if not isinstance(self.retrieved_docs, list) or len(self.retrieved_docs) == 0:
            raise ValueError(f"Retrieved docs must be a non-empty list. Got: {self.retrieved_docs}")
        if not all(isinstance(d, (str, RetrievedDoc)) for d in self.retrieved_docs):
            raise ValueError(
                "Every item in RAGSample.retrieved_docs must be a str or RetrievedDoc. "
                f"Got: {self.retrieved_docs}"
            )
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError(f"RAGSample.answer must be a non-empty string. Got: {self.answer}")

    @property
    def retrieved_texts(self) -> list[str]:
        """Always returns list[str] regardless of whether retrieved_docs is list[str] or list[RetrievedDoc]."""
        return [
            d.content if isinstance(d, RetrievedDoc) else d
            for d in self.retrieved_docs
        ]
