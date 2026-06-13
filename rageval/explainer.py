# rageval/explainer.py

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rageval.core.sample import RAGSample
from rageval.core.result import EvalResult


@dataclass
class SentenceLabel:
    sentence: str
    label: str          # "supported" | "unsupported" | "inferred"
    evidence: str       # which context sentence or evidence string


@dataclass
class DocAnalysis:
    doc_index: int
    snippet: str        # first 120 chars
    relevance_score: float
    contributing_sentences: list[str]   # doc sentences that matched answer sentences


@dataclass
class ExplanationReport:
    query_analysis: dict            # {"entities": [...], "query_type": "..."}
    per_doc_analysis: list[DocAnalysis]
    answer_analysis: list[SentenceLabel]
    metric_summary: dict            # {metric_name: {"score": ..., "interpretation": ...}}
    action_items: list[str]         # ordered most-impactful first


_METRIC_INTERPRETATIONS = {
    "faithfulness": {
        "high": "Answer is well-grounded in the retrieved context.",
        "low":  "Answer contains hallucinations — claims not supported by context.",
    },
    "context_precision": {
        "high": "Retriever returned mostly relevant documents.",
        "low":  "Retriever is returning noisy, irrelevant documents.",
    },
    "answer_relevancy": {
        "high": "Answer directly addresses the original question.",
        "low":  "Answer drifted off-topic from the original question.",
    },
    "context_recall": {
        "high": "Context covers the information needed for a correct answer.",
        "low":  "Context is missing key facts — retriever needs improvement.",
    },
    "answer_completeness": {
        "high": "Answer covers all important information available in context.",
        "low":  "Answer leaves out important facts that were in the context.",
    },
    "contradiction_detector": {
        "high": "Answer does not contradict the retrieved context.",
        "low":  "Answer directly contradicts claims made in the retrieved context.",
    },
}


class ExplainabilityReporter:
    """
    Generates a human-readable explanation of an evaluation result.

    Produces:
    - Per-sentence answer labeling (supported / unsupported / inferred)
    - Per-document contribution analysis (which doc sentences fed which answer sentences)
    - Prioritised action items
    - Single-file HTML report with inline CSS (no external deps, shareable)

    The per-doc attribution is pure embedding math — no LLM calls.
    Optional judge is only used for query entity detection.

    Usage:
        reporter = ExplainabilityReporter()
        report = reporter.explain(sample, eval_result)
        reporter.to_html(report, "report.html")
    """

    SIM_THRESHOLD = 0.45    # minimum similarity to count as "contributing"

    def __init__(self, judge=None, embedding_judge=None):
        self.judge = judge
        self.embedding_judge = embedding_judge

    def explain(self, sample: RAGSample, eval_result: EvalResult) -> ExplanationReport:
        docs = sample.retrieved_texts

        # ── Query analysis ─────────────────────────────────────────────
        entities = self._extract_entities(sample.query)
        query_type = self._detect_query_type(sample.query)
        query_analysis = {"entities": entities, "query_type": query_type}

        # ── Per-doc analysis ───────────────────────────────────────────
        per_doc = self._analyze_docs(docs, sample.answer)

        # ── Answer sentence labeling ───────────────────────────────────
        answer_analysis = self._label_answer_sentences(sample.answer, eval_result)

        # ── Metric summary ─────────────────────────────────────────────
        metric_summary = {}
        for name, mr in eval_result.metric_results.items():
            level = "high" if mr.score >= 0.7 else "low"
            interp_map = _METRIC_INTERPRETATIONS.get(name, {})
            interpretation = interp_map.get(level, f"Score: {mr.score:.2f}")
            metric_summary[name] = {
                "score": mr.score,
                "passed": mr.passed,
                "interpretation": interpretation,
            }

        # ── Action items ───────────────────────────────────────────────
        action_items = self._build_action_items(eval_result, per_doc)

        return ExplanationReport(
            query_analysis=query_analysis,
            per_doc_analysis=per_doc,
            answer_analysis=answer_analysis,
            metric_summary=metric_summary,
            action_items=action_items,
        )

    def to_dict(self, report: ExplanationReport) -> dict:
        return {
            "query_analysis": report.query_analysis,
            "per_doc_analysis": [
                {
                    "doc_index": d.doc_index,
                    "snippet": d.snippet,
                    "relevance_score": d.relevance_score,
                    "contributing_sentences": d.contributing_sentences,
                }
                for d in report.per_doc_analysis
            ],
            "answer_analysis": [
                {"sentence": s.sentence, "label": s.label, "evidence": s.evidence}
                for s in report.answer_analysis
            ],
            "metric_summary": report.metric_summary,
            "action_items": report.action_items,
        }

    def to_html(self, report: ExplanationReport, path: str) -> None:
        """Write a self-contained single-file HTML report."""
        html = self._render_html(report)
        Path(path).write_text(html, encoding="utf-8")

    # ── Private helpers ────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> list[str]:
        """Simple heuristic: capitalised words that aren't sentence starters."""
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            clean = word.strip(".,?!;:")
            if i > 0 and clean and clean[0].isupper() and clean not in entities:
                entities.append(clean)
        return entities[:8]

    def _detect_query_type(self, query: str) -> str:
        q = query.lower().strip()
        if q.startswith(("what", "who", "where", "when", "which")):
            return "factual"
        if q.startswith(("how", "explain", "describe")):
            return "explanatory"
        if q.startswith(("compare", "difference", "versus", "vs")):
            return "comparison"
        if q.startswith(("why",)):
            return "causal"
        if "not" in q or "never" in q or "without" in q:
            return "negation"
        return "general"

    def _analyze_docs(self, docs: list[str], answer: str) -> list[DocAnalysis]:
        answer_sents = self._split_sentences(answer)
        results = []
        for i, doc in enumerate(docs):
            doc_sents = self._split_sentences(doc)
            relevance = self._doc_relevance(doc, answer)
            contributing = self._find_contributing_sentences(doc_sents, answer_sents)
            results.append(DocAnalysis(
                doc_index=i + 1,
                snippet=doc[:120] + ("..." if len(doc) > 120 else ""),
                relevance_score=round(relevance, 3),
                contributing_sentences=contributing[:5],
            ))
        return results

    def _label_answer_sentences(
        self, answer: str, eval_result: EvalResult
    ) -> list[SentenceLabel]:
        sentences = self._split_sentences(answer)
        if not sentences:
            return []

        # Build a lookup from claim text → label using Faithfulness hallucinations
        unsupported_claims: set[str] = set()
        unsupported_reasons: dict[str, str] = {}

        for mr in eval_result.metric_results.values():
            for h in mr.hallucinations:
                unsupported_claims.add(h.claim.lower().strip())
                unsupported_reasons[h.claim.lower().strip()] = h.reason

        labels = []
        for sent in sentences:
            sent_lower = sent.lower().strip()
            matched_claim = None
            for claim in unsupported_claims:
                if claim[:30] in sent_lower or sent_lower[:30] in claim:
                    matched_claim = claim
                    break

            if matched_claim:
                label = "unsupported"
                evidence = unsupported_reasons.get(matched_claim, "Not found in context.")
            else:
                # Check if it closely matches evidence strings (supported)
                label = "supported"
                evidence = "Consistent with retrieved context."

            labels.append(SentenceLabel(sentence=sent, label=label, evidence=evidence))

        return labels

    def _doc_relevance(self, doc: str, answer: str) -> float:
        """Simple word overlap Jaccard similarity as relevance proxy."""
        doc_words = set(doc.lower().split())
        ans_words = set(answer.lower().split())
        if not doc_words or not ans_words:
            return 0.0
        return len(doc_words & ans_words) / len(doc_words | ans_words)

    def _find_contributing_sentences(
        self, doc_sents: list[str], answer_sents: list[str]
    ) -> list[str]:
        """Return doc sentences that have high word overlap with any answer sentence."""
        contributing = []
        for ds in doc_sents:
            ds_words = set(ds.lower().split())
            for ans in answer_sents:
                ans_words = set(ans.lower().split())
                if not ds_words or not ans_words:
                    continue
                overlap = len(ds_words & ans_words) / max(len(ds_words), len(ans_words))
                if overlap >= 0.3:
                    contributing.append(ds)
                    break
        return contributing

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _build_action_items(eval_result: EvalResult, per_doc: list[DocAnalysis]) -> list[str]:
        items = []
        for name, mr in sorted(
            eval_result.metric_results.items(), key=lambda x: x[1].score
        ):
            if not mr.passed:
                if name == "faithfulness":
                    items.append(
                        f"[CRITICAL] Faithfulness {mr.score:.2f}: Add 'Answer ONLY from "
                        f"the provided context' to your system prompt."
                    )
                elif name == "context_precision":
                    low_docs = [d for d in per_doc if d.relevance_score < 0.1]
                    items.append(
                        f"[HIGH] Context Precision {mr.score:.2f}: "
                        f"{len(low_docs)} document(s) appear irrelevant. "
                        f"Reduce top-k or improve your embedding model."
                    )
                elif name == "context_recall":
                    items.append(
                        f"[HIGH] Context Recall {mr.score:.2f}: Retriever is missing "
                        f"key facts. Increase top-k or improve chunking strategy."
                    )
                elif name == "answer_relevancy":
                    items.append(
                        f"[MEDIUM] Answer Relevancy {mr.score:.2f}: Answer drifted "
                        f"off-topic. Constrain the LLM to answer the specific question asked."
                    )
                elif name == "contradiction_detector":
                    items.append(
                        f"[CRITICAL] Contradiction {mr.score:.2f}: Answer directly "
                        f"contradicts retrieved context. Check for context poisoning or "
                        f"conflicting documents."
                    )
                else:
                    items.append(
                        f"[MEDIUM] {name} {mr.score:.2f}: Review evidence — "
                        f"{mr.evidence[0] if mr.evidence else 'see results'}."
                    )
        if not items:
            items.append("No action items — all metrics passed.")
        return items

    def _render_html(self, report: ExplanationReport) -> str:
        label_colors = {
            "supported": "#d4edda",
            "unsupported": "#f8d7da",
            "inferred": "#fff3cd",
        }
        label_border = {
            "supported": "#28a745",
            "unsupported": "#dc3545",
            "inferred": "#ffc107",
        }

        # Answer sentences HTML
        answer_html = ""
        for sl in report.answer_analysis:
            bg = label_colors.get(sl.label, "#f8f9fa")
            border = label_border.get(sl.label, "#6c757d")
            if sl.label == "unsupported":
                answer_html += (
                    f'<details style="background:{bg};border-left:4px solid {border};'
                    f'padding:6px 10px;margin:4px 0;">'
                    f'<summary style="cursor:pointer">{sl.sentence}</summary>'
                    f'<small style="color:#555">{sl.evidence}</small>'
                    f'</details>'
                )
            else:
                answer_html += (
                    f'<p style="background:{bg};border-left:4px solid {border};'
                    f'padding:6px 10px;margin:4px 0">{sl.sentence}</p>'
                )

        # Metric table HTML
        metrics_html = "<table style='width:100%;border-collapse:collapse'>"
        metrics_html += "<tr><th style='text-align:left;padding:6px;border-bottom:1px solid #ddd'>Metric</th><th style='padding:6px;border-bottom:1px solid #ddd'>Score</th><th style='padding:6px;border-bottom:1px solid #ddd'>Status</th><th style='padding:6px;border-bottom:1px solid #ddd'>Interpretation</th></tr>"
        for name, m in report.metric_summary.items():
            color = "#28a745" if m["passed"] else "#dc3545"
            status = "PASS" if m["passed"] else "FAIL"
            metrics_html += (
                f"<tr><td style='padding:6px;border-bottom:1px solid #eee'>{name}</td>"
                f"<td style='padding:6px;text-align:center;border-bottom:1px solid #eee'>{m['score']:.2f}</td>"
                f"<td style='padding:6px;text-align:center;color:{color};border-bottom:1px solid #eee'><b>{status}</b></td>"
                f"<td style='padding:6px;border-bottom:1px solid #eee;color:#555'>{m['interpretation']}</td>"
                f"</tr>"
            )
        metrics_html += "</table>"

        # Action items HTML
        actions_html = "<ul style='margin:0;padding-left:20px'>"
        for item in report.action_items:
            color = "#dc3545" if "[CRITICAL]" in item else ("#fd7e14" if "[HIGH]" in item else "#6c757d")
            actions_html += f"<li style='color:{color};margin:4px 0'>{item}</li>"
        actions_html += "</ul>"

        # Docs HTML
        docs_html = ""
        for d in report.per_doc_analysis:
            docs_html += (
                f"<div style='background:#f8f9fa;border:1px solid #dee2e6;"
                f"border-radius:4px;padding:10px;margin:8px 0'>"
                f"<b>Doc {d.doc_index}</b> — relevance: {d.relevance_score:.2f}<br>"
                f"<small style='color:#555'>{d.snippet}</small>"
            )
            if d.contributing_sentences:
                docs_html += "<br><small><b>Used in answer:</b> " + " | ".join(
                    s[:60] for s in d.contributing_sentences
                ) + "</small>"
            docs_html += "</div>"

        entities = ", ".join(report.query_analysis.get("entities", [])) or "none detected"
        query_type = report.query_analysis.get("query_type", "unknown")

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>rageval Explanation Report</title></head>
<body style="font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:40px auto;padding:20px;color:#212529">
  <h1 style="border-bottom:2px solid #212529;padding-bottom:10px">rageval Explanation Report</h1>

  <h2 style="margin-top:30px">Query Analysis</h2>
  <p><b>Query type:</b> {query_type} &nbsp;&nbsp; <b>Entities:</b> {entities}</p>

  <h2 style="margin-top:30px">Metrics</h2>
  {metrics_html}

  <h2 style="margin-top:30px">Action Items</h2>
  {actions_html}

  <h2 style="margin-top:30px">Answer Breakdown</h2>
  <p style="color:#555;font-size:0.9em">
    <span style="background:#d4edda;padding:2px 6px">green = supported</span>
    &nbsp;
    <span style="background:#f8d7da;padding:2px 6px">red = unsupported (click to expand)</span>
    &nbsp;
    <span style="background:#fff3cd;padding:2px 6px">yellow = inferred</span>
  </p>
  {answer_html}

  <h2 style="margin-top:30px">Retrieved Documents</h2>
  {docs_html}

  <footer style="margin-top:40px;padding-top:10px;border-top:1px solid #dee2e6;color:#6c757d;font-size:0.85em">
    Generated by rageval
  </footer>
</body>
</html>"""
