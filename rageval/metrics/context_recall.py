# rageval/metrics/context_recall.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult

GROUND_TRUTH_CLAIM_EXTRACTION_PROMPT = """\
You are given a ground truth answer. Your job is to extract every individual factual claim.

A claim is a single, self-contained statement that can be independently verified.
Each claim should make sense on its own without needing the rest of the answer.

Ground truth answer:
{ground_truth}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{"claims": ["claim 1", "claim 2", "claim 3"]}}
"""

CONTEXT_COVERAGE_PROMPT = """\
You are given a retrieved context and a list of factual claims from a ground truth answer.
Your job is to determine whether each claim can be found in or inferred from the context.

Rules:
- A claim is "found" if the context explicitly states it or if it can be directly inferred.
- A claim is "not_found" if the context does not contain enough information to support it.
- Do not use your own knowledge. Only use what is in the context.

Context:
{context}

Claims to check:
{claims}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "coverage": [
        {{"claim": "exact claim text", "found": true, "reason": "brief explanation"}},
        {{"claim": "exact claim text", "found": false, "reason": "brief explanation"}}
    ]
}}
"""


class ContextRecall(BaseMetric):
    """
    Measures whether the retrieved context contains all information needed
    to produce the correct answer.

    Score = claims_found_in_context / total_claims_in_ground_truth

    Score of 1.0 = context covers every fact in the ground truth
    Score of 0.0 = context is missing every fact needed to answer correctly

    Requires ground_truth to be set in RAGSample.

    Algorithm:
    1. Extract atomic claims from ground_truth (LLM call 1)
    2. Check each claim against retrieved context (LLM call 2)
    3. Score = found / total
    4. Missing claims -> evidence list
    """

    name = "context_recall"
    required_inputs = ["retrieved_docs", "ground_truth"]

    def score(self, sample: RAGSample) -> MetricResult:
        self.validate(sample)

        context = "\n\n--\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(sample.retrieved_texts)
        )

        # Step 1: extract claims from ground truth
        try:
            extraction = self.judge.complete_json(
                GROUND_TRUTH_CLAIM_EXTRACTION_PROMPT.format(ground_truth=sample.ground_truth)
            )
            claims = extraction.get("claims", [])
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Claim extraction failed: {str(e)}",
                evidence=[],
            )

        if not claims:
            return self._make_result(
                score=1.0,
                reasoning="No factual claims found in ground truth.",
                evidence=[],
            )

        # Step 2: check each claim against the context
        claims_text = "\n".join(f"- {c}" for c in claims)

        try:
            verification = self.judge.complete_json(
                CONTEXT_COVERAGE_PROMPT.format(context=context, claims=claims_text)
            )
            coverage = verification.get("coverage", [])
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Context coverage check failed: {str(e)}",
                evidence=[f"Raw claims extracted: {claims}"],
            )

        if not coverage:
            return self._make_result(
                score=0.0,
                reasoning="Judge returned empty coverage list.",
                evidence=[f"Claims were: {claims}"],
            )

        # Step 3: compute score and build evidence from missing claims
        found = [v for v in coverage if v.get("found", False)]
        missing = [v for v in coverage if not v.get("found", False)]

        score = len(found) / len(coverage)

        evidence = [
            f"MISSING: \"{v.get('claim', '')}\" — {v.get('reason', 'no reason given')}"
            for v in missing
        ]

        if not missing:
            reasoning = f"All {len(coverage)} ground truth claims are present in the retrieved context."
        else:
            reasoning = (
                f"{len(missing)} of {len(coverage)} ground truth claims are missing from "
                f"the retrieved context. See evidence for details."
            )

        return self._make_result(score=score, reasoning=reasoning, evidence=evidence)
