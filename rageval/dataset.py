# rageval/dataset.py

import json
from pathlib import Path
from rageval.judges.base import BaseJudge

QUESTION_GENERATION_PROMPT = """\
You are given a document. Your job is to generate {n} evaluation questions that can be answered using ONLY the information in this document.

Question types to generate: {question_types}

Document:
{document}

Rules:
- Each question must be answerable from the document alone — no outside knowledge needed.
- Make questions specific and varied.
- question_type must be one of: factual, inferential, comparative.

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{
    "questions": [
        {{"query": "question text", "question_type": "factual"}},
        {{"query": "question text", "question_type": "inferential"}}
    ]
}}
"""

GROUND_TRUTH_GENERATION_PROMPT = """\
You are given a document and a question. Your job is to write a complete, accurate ground truth answer using ONLY the information in the document.

Document:
{document}

Question:
{query}

Rules:
- Answer using only facts present in the document.
- Be concise but complete — include all relevant facts.
- Do not add information not present in the document.

Respond ONLY with a JSON object. No explanation before or after. No markdown fences.
{{"answer": "the ground truth answer"}}
"""


class EvalDatasetGenerator:
    """
    Generates evaluation datasets from raw documents.

    The biggest friction point in adopting any eval tool is creating
    evaluation datasets. This generates one in minutes from your knowledge base.

    Each generated question includes:
    - query
    - retrieved_docs  (the source document)
    - answer          (ground truth, LLM-generated)
    - question_type
    - source_doc_index

    Output JSON is directly loadable by `rageval run` — no conversion needed.

    Usage:
        generator = EvalDatasetGenerator(judge=judge)
        questions = generator.generate(documents=my_docs, n_questions=50)
        generator.save(questions, "eval_data.json")
    """

    def __init__(self, judge: BaseJudge):
        self.judge = judge

    def generate(
        self,
        documents: list[str],
        n_questions: int = 20,
        question_types: list[str] = None,
    ) -> list[dict]:
        """
        Generate n_questions evaluation samples from the given documents.

        Questions are distributed evenly across documents. For each question,
        a ground truth answer is generated via a second LLM call.
        """
        if not documents:
            raise ValueError("documents list must not be empty.")
        if question_types is None:
            question_types = ["factual", "inferential", "comparative"]

        questions_per_doc = max(1, n_questions // len(documents))
        remainder = n_questions - questions_per_doc * len(documents)

        all_samples: list[dict] = []

        for doc_idx, document in enumerate(documents):
            n_for_this_doc = questions_per_doc + (1 if doc_idx < remainder else 0)
            types_str = ", ".join(question_types)

            # Step 1: generate questions from this document
            try:
                result = self.judge.complete_json(
                    QUESTION_GENERATION_PROMPT.format(
                        n=n_for_this_doc,
                        question_types=types_str,
                        document=document[:3000],
                    )
                )
                generated = result.get("questions", [])[:n_for_this_doc]
            except Exception:
                # Skip this document on failure rather than aborting the whole run
                generated = []

            # Step 2: generate ground truth answer for each question
            for q in generated:
                query = q.get("query", "").strip()
                q_type = q.get("question_type", "factual")
                if not query:
                    continue

                try:
                    gt_result = self.judge.complete_json(
                        GROUND_TRUTH_GENERATION_PROMPT.format(
                            document=document[:3000],
                            query=query,
                        )
                    )
                    ground_truth = gt_result.get("answer", "").strip()
                except Exception:
                    ground_truth = ""

                all_samples.append({
                    "query": query,
                    "retrieved_docs": [document],
                    "answer": ground_truth,
                    "ground_truth": ground_truth,
                    "question_type": q_type,
                    "source_doc_index": doc_idx,
                })

        return all_samples

    def save(self, questions: list[dict], path: str) -> None:
        """Write generated questions to a JSON file loadable by `rageval run`."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(questions, indent=2), encoding="utf-8")
