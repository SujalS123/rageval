# rageval/metrics/noise_sensitivity.py

import random
from rageval.metrics.base import BaseMetric
from rageval.metrics.faithfulness import Faithfulness
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult


class NoiseSensitivity(BaseMetric):
    """
    Measures how robust the pipeline is when irrelevant documents are injected.

    Algorithm:
    1. Run Faithfulness on the original sample. Store as clean_score.
    2. Inject n_noise random documents from noise_corpus into retrieved_docs and shuffle.
    3. Run Faithfulness again on the noisy sample. Store as noisy_score.
    4. degradation = max(0.0, clean_score - noisy_score)
    5. Score = 1.0 - degradation

    Score of 1.0 = pipeline ignores noise completely (fully robust)
    Score of 0.3 = faithfulness dropped 0.7 when noise was added (fragile pipeline)

    No other RAG evaluation library implements this metric.
    It directly measures whether your pipeline can be manipulated by
    retrieval failures or adversarial inputs.
    """

    name = "noise_sensitivity"
    required_inputs = ["retrieved_docs", "answer"]

    def __init__(self, judge, noise_corpus: list[str], n_noise: int = 2, threshold: float = 0.8):
        super().__init__(judge=judge, threshold=threshold)
        self.noise_corpus = noise_corpus
        self.n_noise = n_noise
        self._faithfulness = Faithfulness(judge=judge, threshold=threshold)

    def score(self, sample: RAGSample) -> MetricResult:
        self.validate(sample)

        # Step 1: clean faithfulness
        try:
            clean_result = self._faithfulness.score(sample)
            clean_score = clean_result.score
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Clean faithfulness run failed: {str(e)}",
                evidence=[],
            )

        # Step 2: build noisy sample
        n = min(self.n_noise, len(self.noise_corpus))
        noise_docs = random.sample(self.noise_corpus, n)
        noisy_retrieved = sample.retrieved_texts + noise_docs
        random.shuffle(noisy_retrieved)

        noisy_sample = RAGSample(
            query=sample.query,
            retrieved_docs=noisy_retrieved,
            answer=sample.answer,
            ground_truth=sample.ground_truth,
            metadata=sample.metadata,
        )

        # Step 3: noisy faithfulness
        try:
            noisy_result = self._faithfulness.score(noisy_sample)
            noisy_score = noisy_result.score
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Noisy faithfulness run failed: {str(e)}",
                evidence=[f"Clean score was: {clean_score:.3f}", f"Noise docs: {noise_docs}"],
            )

        # Step 4 & 5: compute degradation and final score
        degradation = max(0.0, clean_score - noisy_score)
        score = 1.0 - degradation

        reasoning = (
            f"Clean faithfulness: {clean_score:.3f}, noisy faithfulness: {noisy_score:.3f}. "
            f"Degradation: {degradation:.3f} after injecting {n} noise document(s)."
        )

        evidence = [
            f"Clean faithfulness score: {clean_score:.3f}",
            f"Noisy faithfulness score: {noisy_score:.3f}",
            f"Degradation: {degradation:.3f}",
        ] + [f"Noise doc injected: \"{doc}\"" for doc in noise_docs]

        return self._make_result(score=score, reasoning=reasoning, evidence=evidence)
