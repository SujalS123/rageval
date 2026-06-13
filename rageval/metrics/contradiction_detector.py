# rageval/metrics/contradiction_detector.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult

CONTRADICTION_DETECTION_PROMPT = """\
You are given a retrieved context and an answer. Your job is to find any claims in the answer that directly contradict what the context says.

A contradiction is when:
- The answer states X, but the context explicitly states not-X or the opposite of X.
- The answer gives a specific value (number, date, name) that the context gives differently.

A claim is NOT a contradiction if:
- The context simply does not mention it (that is unfaithfulness, not contradiction).
- The answer adds extra detail not in the context.

For each contradiction found, rate severity from 0.0 (minor) to 1.0 (direct factual reversal).

Context:
{context}

Answer:
{answer}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "contradictions": [
        {{
            "claim": "the claim in the answer",
            "context_says": "what the context actually states",
            "reason": "brief explanation of the contradiction",
            "severity": 0.9
        }}
    ]
}}
"""


class ContradictionDetector(BaseMetric):
    """
    Catches the most severe RAG failure: the answer saying the opposite
    of what the context explicitly states.

    Mathematically distinct from Faithfulness:
    - Faithfulness catches claims not present in context.
    - ContradictionDetector catches claims that directly reverse what context states.

    A low faithfulness score could mean the answer has extra information.
    A contradiction score below 1.0 means the LLM actively contradicted what it read.
    These require completely different fixes.

    Score = 1.0 - (contradicted_claims / total_claims_checked)

    Score of 1.0 = no contradictions found
    Score of 0.5 = half the answer claims directly contradict the context

    Does NOT require ground_truth.
    Uses a single LLM call.

    Algorithm:
    1. Send context + answer to LLM, ask it to find direct contradictions
    2. Score = 1.0 - (contradictions / total answer sentences estimated)
    3. Evidence = each contradiction with what context actually says and severity
    """

    name = "contradiction_detector"
    required_inputs = ["retrieved_docs", "answer"]

    def score(self, sample: RAGSample) -> MetricResult:
        self.validate(sample)

        context = "\n\n--\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(sample.retrieved_texts)
        )

        try:
            result = self.judge.complete_json(
                CONTRADICTION_DETECTION_PROMPT.format(
                    context=context,
                    answer=sample.answer,
                )
            )
            contradictions = result.get("contradictions", [])
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Contradiction detection failed: {str(e)}",
                evidence=[],
            )

        if not contradictions:
            return self._make_result(
                score=1.0,
                reasoning="No contradictions found — the answer does not contradict the retrieved context.",
                evidence=[],
            )

        # Estimate total claims as sentences in the answer (simple heuristic)
        # The score is: 1 - (n_contradictions / estimated_total_claims)
        # Clamped so more contradictions than sentences still gives 0.0
        sentence_count = max(len(contradictions), len(sample.answer.split(". ")))
        score = max(0.0, 1.0 - len(contradictions) / sentence_count)

        evidence = [
            f"CONTRADICTION: \"{c.get('claim', '')}\" | "
            f"Context says: \"{c.get('context_says', '')}\" | "
            f"Severity: {float(c.get('severity', 0.5)):.1f} — {c.get('reason', '')}"
            for c in contradictions
        ]

        reasoning = (
            f"{len(contradictions)} direct contradiction(s) found between the answer "
            f"and the retrieved context. See evidence for details."
        )

        return self._make_result(score=score, reasoning=reasoning, evidence=evidence)
