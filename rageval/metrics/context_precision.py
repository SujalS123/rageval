#rageval/metrics/context_precison.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult
from rageval.core.retrieved_doc import RetrievedDoc

RELEVANCE_PROMPT = """\
You are evaluating whether a retrived document is useful for answering a question.

Question: {query}

Retrieved Document:
{doc}

Determine: did this specific document contribute information toward answering the question?
A document is useful it is contains facts , context ,or evidence that helps answer the question.
A document is Not useful it is about a completely different topic.

Think step by step. Then respond ONLY with a JSON object. No explanation. No markdown fences.
{{"is_relevant":true , "reason":"brief explanation of why useful or not"}} 
"""

class ContextPrecision(BaseMetric):
    """
        Measures what fraction of retrieved documents were actually useful.

        Score = useful_documents / total_retrieved_documents

        Score of 1.0 = every retrived document helped answer the question
        Score of 0.3 = only 30% of retrieved documents were relevant (noisy retriever)
        Score of 0.0 = none of the retrieved documents were relevant (broken retriever)

        What a low score tells you;
        -Your retriever is returuning too many irrelevant chunks
        -Fix by: reducing top-k , improving embeddingd , better chunking strategy

        Does NOT require groun_truth.
        Makes one LLM call per retrieved document.

        Algorithm:
        1. For each retrieved document, call LLM to judge relevance
        2. Score = sum of relevant / total retrieved
        3. Evidence = list of irrelevant documents with source when available
    """
    name = "context_precision"
    required_inputs = ["query", "retrieved_docs"]

    def score(self, sample: RAGSample) -> MetricResult:
        self.validate(sample)

        relevance_results = []

        for i, raw_doc in enumerate(sample.retrieved_docs):
            # Extract text and source — works for both str and RetrievedDoc
            if isinstance(raw_doc, RetrievedDoc):
                text = raw_doc.content
                source = raw_doc.source
            else:
                text = raw_doc
                source = None

            try:
                result = self.judge.complete_json(
                    RELEVANCE_PROMPT.format(
                        query=sample.query,
                        doc=text[:1500],
                    )
                )
                is_relevant = result.get("is_relevant", False)
                reason = result.get("reason", "no reason given")

            except Exception as e:
                is_relevant = False
                reason = f"evaluation failed: {str(e)}"

            relevance_results.append({
                "doc_index": i + 1,
                "is_relevant": is_relevant,
                "reason": reason,
                "source": source,
                "snippet": text[:80] + "..." if len(text) > 80 else text,
            })

        relevant = [r for r in relevance_results if r["is_relevant"]]
        irrelevant = [r for r in relevance_results if not r["is_relevant"]]

        score = len(relevant) / len(relevance_results) if relevance_results else 0.0

        # Include source in evidence when available
        evidence = []
        for r in irrelevant:
            if r["source"] is not None:
                evidence.append(
                    f"Doc {r['doc_index']} NOT USEFUL (source: {r['source']}): {r['reason']}"
                )
            else:
                evidence.append(
                    f"Doc {r['doc_index']} NOT USEFUL: {r['reason']} | \"{r['snippet']}\""
                )

        if not irrelevant:
            reasoning = f"All {len(relevance_results)} retrieved documents were relevant to the query."
        else:
            reasoning = (
                f"{len(irrelevant)} of {len(relevance_results)} retrieved documents "
                f"were not useful for answering the query."
            )

        return self._make_result(
            score=score,
            reasoning=reasoning,
            evidence=evidence,
        )
