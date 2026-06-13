# rageval/metrics/answer_relevancy.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult
from rageval.judges.heuristic import HeuristicJudge

REVERSE_QUESTION_PROMPT ="""\
You are given an answer. Your job is to generate questions that this answer would be
a good reasponse to.

Answer:
{answer}

Generate exactly 3 different questions that this answer appropriately addresses.
The questions should vary in phrasing but all capture the same intent.

Respond ONLY with a JSON object. NO explanation.No markdwon fences.
{{"questions":["question 1","question 2","question 3"]}}
"""

class AnswerRelevancy(BaseMetric):
    """
    Measures whether the anser actually addresses the original query.
    
    Uses reverse inference - no ground truth needed.
    
    Algorithm:
    1. Given the answer, ask LLM: what questions would this answer address?
    2. Generate 3 candidate questions
    3. Compute cosine similarity between original query and each candidate
    4. Average similarity = relevancy score
    
    Intutition:if the answer is truly relevant , reverse-engineering the question
    from it should reproduce somthings close to the original query.
    An off-topic answer will suggest completely different questions.
    
    Score of 1.0 = answaer directly addresses the query
    Score of 0.3 = answer drifted completely off topic
    
    Uses two judges:
    -LLM judge: genrates the reverse questions
    -HeuristicJudge: computes embedding similarity (no extra API cost)
    """

    name =  "answer_relevancy"
    requires_inputs = ["query" , "answer"]

    def __init__(
            self,
            judge,
            threshold: float = 0.7 , 
            embedding_judge: HeuristicJudge = None,
    ):
        super().__init__(judge=judge , threshold=threshold)
        #Create a default embedding judge if none provided
        self.embedding_judge = embedding_judge or HeuristicJudge()

    
    def score(self , sample:RAGSample) -> MetricResult:
        self.validate(sample)

        # Step 1: Generate reverse questions from the answer
        try:
            result = self.judge.complete_json(
                REVERSE_QUESTION_PROMPT.format(answer=sample.answer)
            )
            generated_questions = result.get("questions",[])

        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Reverse question generation failed: {str(e)}",
                evidence=[]
            )
        
        # Step 2: Compute cosine similarity between original quey 
        # and each generated question using embeddings (free , no API call)
        similarities = self.embedding_judge.batch_similarity(
            anchor=sample.query,
            texts=generated_questions,
        )

        if not similarities:
            return self._make_result(
                score = 0.0 ,
                reasoning = "Could not compute similarity scores.",
                evidence = [],
            )
        avg_similarity = sum(similarities)/len(similarities)

        # Evidence = the generated questions with their similarity scores
        # Low similarity questions show why the answer was off-topic
        evidence = [
            f"Generated Q{i+1}: \"{q}\" -. similarity: {s:.2f}"
            for i , (q,s) in enumerate(zip(generated_questions , similarities))
        ]

        reasoning = (
            f"Average cosine similarity between original quey and "
            f"{len(generated_questions)} reverse-generated questions: {avg_similarity:.2f}"
        )

        return self._make_result(
            score=avg_similarity,
            reasoning=reasoning,
            evidence=evidence,
        )