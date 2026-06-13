# rageval/metrics/answer_completeness.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult

RELEVANT_FACTS_EXTRACTION_PROMPT = """\
You are given a query and a retrieved context. Your job is to extract every fact in the context that is relevant to answering the query.

A fact is relevant if knowing it would help answer the query more completely or accurately.
Only extract facts that are directly present in the context — do not infer or add outside knowledge.

Query:
{query}

Context:
{context}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{"facts": ["fact 1", "fact 2", "fact 3"]}}
"""

FACT_COVERAGE_PROMPT = """\
You are given a list of relevant facts and an answer. Your job is to determine whether each fact is mentioned or conveyed in the answer.

Rules:
- A fact is "mentioned" if the answer explicitly states it or clearly conveys the same information.
- A fact is "missing" if the answer does not include it at all.
- Do not penalize for phrasing differences — only check whether the information is present.

Answer:
{answer}

Facts to check:
{facts}

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "coverage": [
        {{"fact": "exact fact text", "mentioned": true, "reason": "brief explanation"}},
        {{"fact": "exact fact text", "mentioned": false, "reason": "brief explanation"}}
    ]
}}
"""


class AnswerCompleteness(BaseMetric):
    """
    Measures whether the answer covers all important information available
    in the context that is relevant to the query.

    Faithfulness catches what the answer adds that is wrong.
    Completeness catches what the answer leaves out that is important.

    Score = relevant facts mentioned in answer / total relevant facts in context

    Score of 1.0 = answer covers everything important in the context
    Score of 0.4 = answer only mentioned 40% of the available relevant information

    Does NOT require ground_truth.

    Algorithm:
    1. Extract all facts relevant to the query from the retrieved context (LLM call 1)
    2. Check which facts the answer mentions (LLM call 2)
    3. Score = mentioned / total
    4. Missing facts -> evidence list
    """

    name = "answer_completeness"
    required_inputs = ["query", "retrieved_docs", "answer"]

    def score(self, sample: RAGSample) -> MetricResult:
        self.validate(sample)

        context = "\n\n--\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(sample.retrieved_texts)
        )

        # Step 1: extract relevant facts from context
        try:
            extraction = self.judge.complete_json(
                RELEVANT_FACTS_EXTRACTION_PROMPT.format(
                    query=sample.query,
                    context=context,
                )
            )
            facts = extraction.get("facts", [])
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Fact extraction failed: {str(e)}",
                evidence=[],
            )

        if not facts:
            return self._make_result(
                score=1.0,
                reasoning="No relevant facts found in context for this query.",
                evidence=[],
            )

        # Step 2: check which facts the answer mentions
        facts_text = "\n".join(f"- {f}" for f in facts)

        try:
            verification = self.judge.complete_json(
                FACT_COVERAGE_PROMPT.format(
                    answer=sample.answer,
                    facts=facts_text,
                )
            )
            coverage = verification.get("coverage", [])
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Fact coverage check failed: {str(e)}",
                evidence=[f"Extracted facts: {facts}"],
            )

        if not coverage:
            return self._make_result(
                score=0.0,
                reasoning="Judge returned empty coverage list.",
                evidence=[f"Facts were: {facts}"],
            )

        # Step 3: compute score and build evidence from missing facts
        mentioned = [v for v in coverage if v.get("mentioned", False)]
        missing = [v for v in coverage if not v.get("mentioned", False)]

        score = len(mentioned) / len(coverage)

        evidence = [
            f"MISSING FROM ANSWER: \"{v.get('fact', '')}\" — {v.get('reason', 'no reason given')}"
            for v in missing
        ]

        if not missing:
            reasoning = f"Answer covers all {len(coverage)} relevant facts from the context."
        else:
            reasoning = (
                f"{len(missing)} of {len(coverage)} relevant facts from the context "
                f"were not mentioned in the answer."
            )

        return self._make_result(score=score, reasoning=reasoning, evidence=evidence)
