#ragas/metrics/faithfulness.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult
from rageval.core.hallucination import Hallucination, HallucinationType

CLAIM_EXTRACTION_PROMPT = """\
You are given an answer text. Your job is to extract every individual factual claim \
that makes an assertion about the real world.

A claim is a single, self-contained statement that can be independently verified \
against external evidence. Each claim should make sense on its own.

IMPORTANT rules:
- Only extract claims that assert real-world facts (about people, places, events, \
  science, history, etc.).
- Do NOT extract meta-statements about the answer itself (e.g. do NOT produce \
  claims like "the answer is X" or "the text says Y").
- Do NOT extract greetings, filler phrases, or statements of uncertainty.
- If the answer contains NO verifiable real-world factual claims, return an empty list.

Answer:
{answer}

Think step by step:
1. Read the answer carefully.
2. Identify every distinct real-world factual assertion.
3. Write each as a complete, standalone sentence.
4. If no such claims exist, return an empty list.

Examples:
- Answer: "Paris is the capital of France and has a population of 2 million."
  Claims: ["Paris is the capital of France", "Paris has a population of 2 million."]
- Answer: "I don't know."
  Claims: []
- Answer: "test answer"
  Claims: []

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{"claims": ["claim 1", "claim 2", "claim 3"]}}
"""

CLAIM_VERIFICATION_PROMPT = """\
You are given a context and a list of claims. Your job is to determine whether each
is supported by the context.

Rules:
- A claim is "supported" if the context explicitly states it or if it can be directly inferred.
- A claim is "not_supported" if the context does not mention it, contradicts it,
or if it requires outside knowledge not present in the context.
- Do not use your own knowledge. Only use what is in the context.
- For every unsupported claim, classify it into exactly one type:
    "factual_error"       — the claim states something verifiably false
    "unsupported_claim"   — the claim states something not present in context
    "contradiction"       — the claim directly contradicts the context
    "fabricated_detail"   — the claim invents specific numbers, names, or dates
- Rate severity from 0.0 (minor) to 1.0 (critical) for each unsupported claim.

Context:
{context}

Claims to verify:
{claims}

Think step by step for each claim before deciding.

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "verifications": [
        {{"claim": "exact claim text", "supported": true, "reason": "brief explanation", "type": null, "severity": null}},
        {{"claim": "exact claim text", "supported": false, "reason": "brief explanation", "type": "unsupported_claim", "severity": 0.7}}
    ]
}}
"""

class Faithfulness(BaseMetric):
    """
    Measures whether the anser makes claims supoorted by the restrived context.
    
    Score = supported_claims / total_claims
    
    Score of 1.0 = every claim in the answer is grponded in the context
    Score of 0.0 = no claims are supported (complete hallucination)
    Score of 0.6 = 605 of claims are supported , 40% are hallucinated
    
    Does Not require gorund_truth
    
    Algorithm:
    1. Extract atomic claims from answer (LLM call 1)
    2. Verify each against the context (LLM call 2)
    3. Score = supported / total
    4. Unsupported claims -> evidence list(your debug output)
    
    Why two LLM calls: separating extraction from verification gives the LLM oone focused task per call , producing more accurate results
    than asking both in one call.
    """

    name = "faithfulness"
    required_inputs = ["retrieved_docs" , "answer"]

    def score(self , sample: RAGSample) -> MetricResult:
        self.validate(sample)

        #join all retrieved docs into one context string
        #Number them so verification results are traceable

        context = "\n\n--\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(sample.retrieved_texts)
        )

        #_----Step 1 : Extract claims from the answer ----------
        try:
            extraction = self.judge.complete_json(
               CLAIM_EXTRACTION_PROMPT.format(answer = sample.answer) 
            )
            claims = extraction.get("claims",[])
        except Exception as e:
            return self._make_result(
                score = 0.0,
                reasoning = f"Claim extraction failed: {str(e)}",
                evidence = []
            )
        
        # Edge case: answer has no factual claims (e.g. "I don't Know")
        if not claims:
            return self._make_result(
                score = 1.0,
                reasoning = "No factual claims found in the answer.",
                evidence = [],
            )
        
        #---------Step 2: Verify each claim against the context --------
        claims_text = "\n".join(f"- {c}" for c in claims)

        try:
            verification = self.judge.complete_json(
                CLAIM_VERIFICATION_PROMPT.format(
                    context = context,
                    claims = claims_text,
                )
            )
            verifications = verification.get("verifications",[])
        except Exception as e:
            return self._make_result(
                score = 0.0 ,
                reasoning = f"Claim verfication failed: {str(e)}",
                evidence = [f"Raw claims extracted: {claims}"],
            )
        
        if not verifications:
            return self._make_result(
                score=0.0,
                reasoning="Judge returned empty verification list.",
                evidence=[f"Claims extracted: {claims}"],
            )

        #-------step 3: compute score, build Hallucination objects, derive evidence -------
        supported = [v for v in verifications if v.get("supported", False)]
        unsupported = [v for v in verifications if not v.get("supported", False)]

        score = len(supported) / len(verifications) if verifications else 0.0

        # Build structured Hallucination objects for each unsupported claim
        hallucinations = []
        for v in unsupported:
            raw_type = v.get("type") or "unsupported_claim"
            try:
                h_type = HallucinationType(raw_type)
            except ValueError:
                h_type = HallucinationType.UNSUPPORTED_CLAIM
            hallucinations.append(Hallucination(
                claim=v.get("claim", ""),
                type=h_type,
                severity=float(v.get("severity") or 0.5),
                reason=v.get("reason", "no reason given"),
            ))

        # Evidence is derived from Hallucination objects — backward compatible plain strings
        evidence = [
            f"{h.type.value.upper()}: '{h.claim}' (severity: {h.severity:.1f}) — {h.reason}"
            for h in hallucinations
        ]

        if not unsupported:
            reasoning = (
                f"All {len(verifications)} claims are supported by the reterived context."
            )
        else:
            reasoning = (
                f"{len(unsupported)} of {len(verifications)} claims could not be "
                f"verified from the context. See evidence for details."
            )

        return self._make_result(
            score=score,
            reasoning=reasoning,
            evidence=evidence,
            hallucinations=hallucinations,
        )