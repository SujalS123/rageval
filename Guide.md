# rageval — Complete Agent Reference Guide

---

## Problem Statement

Large language model pipelines that use Retrieval-Augmented Generation (RAG) fail in ways that are invisible without evaluation. A chatbot can return confident, well-written answers that are completely fabricated. A retriever can return 10 documents where 8 are irrelevant noise. An answer can be grammatically perfect while ignoring the actual question.

Existing evaluation tools — RAGAS, DeepEval, TruLens — solve part of this problem. They return a numeric score. `faithfulness: 0.43`. That number tells you something is wrong. It does not tell you what is wrong, which sentence hallucinated, which document was irrelevant, or what to fix.

Teams using these tools end up reading hundreds of outputs manually to find failures. There is no way to gate a deployment on evaluation quality without building custom tooling. There is no debug output to share with a team when scores drop.

**rageval was built to solve this exactly.** Every score comes with evidence — the specific claims that failed, the specific documents that were irrelevant, the specific reasons the answer drifted off topic. Not a number. A diagnosis.

---

## What rageval Is

rageval is a Python library for evaluating RAG pipelines. It is:

- **Framework-agnostic** — takes plain Python strings, not LangChain or LlamaIndex objects
- **Debug-first** — every metric returns evidence explaining why the score is what it is
- **Judge-agnostic** — works with OpenAI, Anthropic, Ollama, or any LLM via one adapter class
- **CI-native** — CLI exits with code 1 on failure, blocking deployments when quality drops
- **Zero lock-in** — install with pip, use with any RAG stack, remove without refactoring

The core input is a `RAGSample` — four plain Python fields:

```python
RAGSample(
    query="What caused the 2008 financial crisis?",
    retrieved_docs=["The crisis was triggered by...", "Lehman Brothers filed..."],
    answer="The crisis was caused by mortgage-backed securities and alien intervention.",
    ground_truth=None,  # optional
)
```

The core output is an `EvalResult` with per-metric `MetricResult` objects:

```python
MetricResult(
    metric_name="faithfulness",
    score=0.67,
    passed=False,
    reasoning="1 of 3 claims could not be verified from context.",
    evidence=[
        'NOT SUPPORTED: "alien intervention" — not mentioned in any retrieved document'
    ],
    threshold=0.8,
)
```

---

## Architecture

```
rageval/
├── core/
│   ├── sample.py         RAGSample dataclass — the input contract
│   ├── result.py         MetricResult + EvalResult — the output contract
│   ├── pipeline.py       evaluate() + batch_evaluate() + summary()
│   ├── retrieved_doc.py  RetrievedDoc with source tracking
│   └── hallucination.py  Hallucination + HallucinationType
├── metrics/
│   ├── base.py                   BaseMetric ABC
│   ├── faithfulness.py
│   ├── context_precision.py
│   ├── answer_relevancy.py
│   ├── context_recall.py
│   ├── noise_sensitivity.py
│   ├── answer_completeness.py
│   └── contradiction_detector.py
├── judges/
│   ├── base.py           BaseJudge ABC — adapter pattern for LLM providers
│   ├── openai_judge.py
│   ├── anthropic_judge.py
│   ├── gemini_judge.py
│   ├── ollama_judge.py
│   ├── cohere_judge.py
│   ├── groq_judge.py
│   └── heuristic.py      Embedding-based judge — free, no API calls
├── reporters/
│   └── json_csv.py       JSON and CSV export
├── tracker.py            RunTracker — SQLite regression tracking
├── dataset.py            EvalDatasetGenerator
├── trace.py              RAGTracer — step-level pipeline tracing
├── autoeval.py           AutoEval — production sampling decorator
├── query_classifier.py   QueryClassifier — query type analysis
├── chunk_analyzer.py     ChunkQualityAnalyzer — pre-indexing chunk analysis
├── consistency.py        ConsistencyAnalyzer — paraphrase consistency testing
├── mcp_server.py         MCP server for AI coding assistant integration
└── cli.py                rageval run / history / diff / traces / trace / init / init-ci / serve
```

### How the layers connect

```
User calls evaluate(sample, metrics)
         ↓
pipeline.py — loops metrics, catches errors, computes weighted score
         ↓
metric.score(sample) — each metric runs independently
         ↓
judge.complete_json(prompt) — sends prompt to LLM, returns parsed JSON
         ↓
MetricResult — score + reasoning + evidence returned up the chain
         ↓
EvalResult — all MetricResults aggregated into one object
```

### The adapter pattern in judges

Every judge inherits `BaseJudge` and implements one method: `complete()`. Metrics never import `anthropic` or `openai` directly — they only talk to `BaseJudge`. Swapping providers means changing one line for the user, touching zero metric code.

```python
class BaseJudge(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str: ...
    def complete_json(self, prompt: str, system: str = "") -> dict: ...
    # complete_json handles JSON parsing with 3-attempt fallback
```

### The ABC pattern in metrics

Every metric inherits `BaseMetric` and implements `score()`. The `@abstractmethod` decorator means Python raises `TypeError` at instantiation if `score()` is missing — catching missing implementations at startup, not during evaluation runs.

---

## What Is Built (v0.4.0)

### RAGSample

```python
@dataclass
class RAGSample:
    query: str
    retrieved_docs: list[str | RetrievedDoc]  # accepts both types
    answer: str
    ground_truth: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def retrieved_texts(self) -> list[str]: ...  # always returns plain strings
```

Validates on construction. Raises `ValueError` immediately if query, docs, or answer are empty. Accepts either `list[str]` or `list[RetrievedDoc]` in `retrieved_docs` — fully backward compatible.

---

### RetrievedDoc

```python
@dataclass
class RetrievedDoc:
    content: str
    source: str = "unknown"   # "vector", "bm25", "graph", "web"
    score: float = 1.0        # retrieval confidence score
    doc_id: str = ""          # optional document identifier
```

Optional upgrade to `retrieved_docs`. Enables source-aware evidence output in ContextPrecision: `Doc 2 NOT USEFUL (source: bm25): reason here`. All metrics use `sample.retrieved_texts` internally and work with both `str` and `RetrievedDoc` transparently.

---

### MetricResult

```python
@dataclass
class MetricResult:
    metric_name: str
    score: float               # 0.0 to 1.0
    passed: bool               # score >= threshold
    reasoning: str             # WHY the score is this value
    evidence: list[str]        # WHICH specific things caused the score
    threshold: float
    hallucinations: list = []  # list[Hallucination] — populated by Faithfulness
```

The `evidence` field is the differentiator. Every other tool stops at `score`. rageval surfaces the specific sentences, claims, and documents that drove the score down. The `hallucinations` field is optional — all non-Faithfulness metrics leave it empty.

---

### HallucinationClassifier

```python
class HallucinationType(str, Enum):
    FACTUAL_ERROR = "factual_error"         # states something verifiably false
    UNSUPPORTED_CLAIM = "unsupported_claim" # states something not in context
    CONTRADICTION = "contradiction"          # contradicts the context directly
    FABRICATED_DETAIL = "fabricated_detail" # invents specific numbers or names

@dataclass
class Hallucination:
    claim: str
    type: HallucinationType
    severity: float  # 0.0 to 1.0
    reason: str
```

Built into `Faithfulness`. The verification prompt now asks the judge to classify each unsupported claim into one of the four types and rate severity. Evidence strings are derived from `Hallucination` objects — backward compatible with all code reading `evidence` as strings. Teams can also read `result.hallucinations` directly to filter by type or severity.

**Example evidence output:**
```
UNSUPPORTED_CLAIM: 'Revenue grew 40%' (severity: 0.8) — context says 12%
CONTRADICTION: 'The treaty was signed in 1820' (severity: 1.0) — context explicitly states 1847
FABRICATED_DETAIL: 'designed by Leonardo da Vinci' (severity: 0.9) — context says Gustave Eiffel
```

---

### Faithfulness

**Measures:** does the answer make claims that cannot be verified from the retrieved context?

**Algorithm:**
1. Extract atomic claims from the answer (LLM call 1)
2. Verify each claim against retrieved context, classify type and severity (LLM call 2)
3. Score = supported claims / total claims

**Math:**
```
Faithfulness = |supported_claims| / |total_claims|
```

**Evidence output:** structured `Hallucination` objects, serialized to evidence strings

**Does NOT need ground truth**

---

### ContextPrecision

**Measures:** of all chunks retrieved, what fraction were actually useful?

**Algorithm:**
1. For each retrieved document, ask judge: is this relevant to the query?
2. Score = relevant documents / total retrieved documents

**Math:**
```
Context Precision = |useful_chunks| / |total_retrieved_chunks|
```

**Evidence output:** the irrelevant documents with reasons; includes source when `RetrievedDoc` is used

**Does NOT need ground truth**

**What a low score tells you:** your retriever is returning noise. Fix by reducing top-k, improving embeddings, or improving chunking strategy.

**Example evidence:**
```
Doc 3 NOT USEFUL (source: bm25): About French cuisine, unrelated to the query about economics
Doc 3 NOT USEFUL: About French cuisine, unrelated to the query about economics  ← plain str input
```

---

### AnswerRelevancy

**Measures:** does the answer actually address the original question?

**Algorithm:**
1. Given the answer, ask LLM to generate 3 questions this answer would address
2. Compute cosine similarity between original query and each generated question
3. Score = average cosine similarity

**Math:**
```
AnswerRelevancy = (1/N) × Σ cosine_similarity(original_query, generated_question_i)
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

**Evidence output:** all generated questions with their similarity scores

**Does NOT need ground truth**

**Why reverse generation:** directly asking "is this relevant?" produces inconsistent scores. Reverse generation forces the LLM to commit to what question the answer addresses, which is easier to score accurately.

---

### ContextRecall

**Measures:** does the retrieved context contain all information needed to produce the correct answer?

**Algorithm:**
1. Break the ground truth answer into atomic factual claims (LLM call 1)
2. For each claim, check whether it can be found in or inferred from retrieved context (LLM call 2)
3. Score = claims found in context / total claims in ground truth

**Math:**
```
Context Recall = |claims_found_in_context| / |total_claims_in_ground_truth|
```

**Evidence output:** the specific claims that were NOT found in context — exactly what the retriever missed

**Requires ground_truth**

**What a low score tells you:** your retriever is missing important documents. Fix chunking strategy, embedding model, or increase top-k.

**Example evidence:**
```
MISSING: "The treaty was ratified by 12 nations" — context does not mention ratification
MISSING: "Revenue grew 40% in Q3" — context only mentions annual figures
```

---

### NoiseSensitivity

**Measures:** how robust is the pipeline when irrelevant documents are injected into context?

**Algorithm:**
1. Run Faithfulness on the original clean context. Store as `clean_score`.
2. Inject `n_noise` random documents from a `noise_corpus` into `retrieved_docs` and shuffle.
3. Run Faithfulness again on the noisy sample. Store as `noisy_score`.
4. `degradation = max(0.0, clean_score - noisy_score)`
5. `score = 1.0 - degradation`

**Math:**
```
NoiseSensitivity = 1.0 - max(0.0, clean_faithfulness - noisy_faithfulness)
Score of 1.0 = pipeline ignores noise completely (fully robust)
Score of 0.3 = faithfulness dropped 0.7 when noise was added (fragile pipeline)
```

**No other RAG evaluation library implements this metric.** It directly measures whether your pipeline can be manipulated by retrieval failures or adversarial inputs.

**Constructor:** `NoiseSensitivity(judge, noise_corpus: list[str], n_noise: int = 2, threshold: float = 0.8)`

**Evidence output:** clean score, noisy score, degradation amount, and the specific noise documents injected

---

### AnswerCompleteness

**Measures:** does the answer cover all important information available in the context relevant to the query?

The complement of Faithfulness — faithfulness catches what the answer *adds* that is wrong; completeness catches what the answer *leaves out* that is important.

**Algorithm:**
1. Extract all facts in the context relevant to the query (LLM call 1)
2. Check which facts the answer mentions (LLM call 2)
3. Score = mentioned facts / total relevant facts

**Evidence output:** the specific facts available in context that the answer omitted

**Does NOT require ground_truth**

**Example evidence:**
```
MISSING FROM ANSWER: "The policy includes a 30-day grace period" — not mentioned in answer
MISSING FROM ANSWER: "Exceptions apply to non-profit organizations" — not mentioned in answer
```

---

### ContradictionDetector

**Measures:** does the answer directly contradict what the retrieved context states?

Mathematically distinct from Faithfulness. Faithfulness catches claims not in context. ContradictionDetector catches claims that directly reverse what context states. These require different fixes.

**Algorithm:**
1. Send context + answer to LLM, ask it to find direct contradictions (single LLM call)
2. Score = 1.0 − (contradictions / total answer claims)

**Evidence output:** each contradiction with the contradicting claim, what context actually says, and severity

**Does NOT require ground_truth**

**Example evidence:**
```
CONTRADICTION: "The policy was approved." | Context says: "The policy was rejected." | Severity: 1.0
```

---

### ConsistencyAnalyzer

Measures whether the pipeline gives consistent answers across semantically equivalent queries (paraphrases).

```python
from rageval import ConsistencyAnalyzer
from rageval.judges.heuristic import HeuristicJudge

analyzer = ConsistencyAnalyzer(judge=judge, embedding_judge=HeuristicJudge())

def my_pipeline(query):
    docs = retriever.search(query)
    answer = llm.generate(query, docs)
    return docs, answer

report = analyzer.analyze(
    query="What caused the 2008 crisis?",
    paraphrases=["Why did the 2008 financial crisis happen?",
                 "What were the causes of the 2008 recession?"],
    pipeline_fn=my_pipeline,
)
print(report.summary())
```

**Algorithm:**
1. Run `pipeline_fn` on original query and all paraphrases
2. Extract atomic claims from each answer (one LLM call per answer)
3. Cross-compare every pair of answers for contradictions and inconsistencies (one LLM call per pair)
4. `consistency_score = 1.0 - (contradictions / total_claim_comparisons)`
5. Root cause: if `embedding_judge` shows paraphrases retrieved different documents, root cause is vocabulary mismatch

**ConsistencyReport fields:** `consistency_score`, `answer_similarity_scores`, `inconsistencies` (list of `ConsistencyItem`), `root_cause_hypothesis`, `fix_suggestion`

**Score of 1.0** = pipeline gives identical factual claims regardless of how the question is phrased
**Score of 0.5** = half the claim comparisons resulted in direct contradictions — users get different facts depending on phrasing

**Root cause diagnosis:**
- Low doc similarity between paraphrases → vocabulary mismatch in embedding model → fix with query expansion or domain-tuned embeddings
- High doc similarity but contradicting answers → generation instability → fix with stricter system prompt and temperature=0

---

### RAGTracer

Persists named evaluation runs to a local SQLite database. Zero new dependencies — `sqlite3` is in Python's standard library.

```python
from rageval import RunTracker

tracker = RunTracker()  # creates .rageval/runs.db
tracker.save_run("v2.3-deploy", results)
tracker.compare_runs("v2.2-deploy", "v2.3-deploy")
tracker.list_runs()
```

CLI commands: `rageval history`, `rageval diff RUN_A RUN_B`, `--save-run` flag on `rageval run`.

Stores: run name, timestamp, total samples, overall score, pass rate, per-metric averages, hallucination type counts.

---

### EvalDatasetGenerator

Generates evaluation datasets from raw documents. Removes the biggest friction point in RAG evaluation adoption.

```python
from rageval import EvalDatasetGenerator

generator = EvalDatasetGenerator(judge=judge)
questions = generator.generate(documents=my_docs, n_questions=50)
generator.save(questions, "eval_data.json")  # ready for rageval run
```

Two-prompt pipeline per question: generate question from document, then generate ground truth answer. Output JSON is directly loadable by `rageval run` with no conversion.

---

### GitHub Actions Generator + Project Init

```bash
rageval init          # creates starter eval_data.json + rageval.md
rageval init-ci       # generates .github/workflows/rag-eval.yml
```

`init-ci` detects the judge from `--judge` flag and sets the correct API key secret in the workflow. One command from zero to CI/CD evaluation.

---

### evaluate()

```python
def evaluate(
    sample: RAGSample,
    metrics: list[BaseMetric],
    weights: dict[str, float] = None,
) -> EvalResult:
```

Runs all metrics. Never raises — errors are stored in `MetricResult.reasoning`. Computes weighted overall score.

---

### batch_evaluate()

```python
def batch_evaluate(
    samples: list[RAGSample],
    metrics: list[BaseMetric],
    max_workers: int = 4,
    show_progress: bool = True,
) -> list[EvalResult]:
```

Parallel evaluation using `ThreadPoolExecutor`. Results returned in same order as input.

---

### CLI

```bash
rageval run eval_data.json --judge openai --model gpt-4o-mini --threshold 0.8
rageval run eval_data.json --judge groq --model llama-3.1-8b-instant --threshold 0.8
rageval run eval_data.json --judge ollama --model llama3 --threshold 0.8
```

Exits with code 1 if overall pass rate is below threshold. This is the CI/CD integration — one line in a GitHub Actions workflow blocks deployments when quality drops.

---

### Judges

| Judge | Provider | Cost | Install | Use case |
|---|---|---|---|---|
| `OpenAIJudge` | OpenAI API | Per token | built-in | Production evaluation |
| `AnthropicJudge` | Anthropic API | Per token | built-in | Production evaluation |
| `GeminiJudge` | Google Gemini API | Per token | `pip install rageval-core[gemini]` | Production evaluation |
| `CohereJudge` | Cohere API | Per token | `pip install rageval-core[cohere]` | Production evaluation |
| `GroqJudge` | Groq API | Free tier | `pip install rageval-core[groq]` | Development / CI pipelines |
| `OllamaJudge` | Local Ollama | Free | `pip install rageval-core[ollama]` | Air-gapped / private evaluation |
| `HeuristicJudge` | Local embeddings | Free | built-in | Development iteration |

---

## What Is Not Built Yet (Roadmap)

### Priority 1 — Advanced RAG support

**AgenticRAGSample**

For pipelines that make multiple retrieval calls — LangGraph, AutoGen, CrewAI.

```python
@dataclass
class AgentStep:
    step_number: int
    tool_used: str
    query_used: str
    retrieved_docs: list[str]
    intermediate_answer: str

@dataclass
class AgenticRAGSample:
    original_query: str
    steps: list[AgentStep]
    final_answer: str
    ground_truth: Optional[str] = None
```

Evaluates each step independently AND the final answer. Identifies which step in a multi-hop chain caused the failure.

---

**GraphRAG support**

Current `RAGSample` works with Graph RAG by stringifying nodes:

```python
# Convert graph nodes to strings
graph_nodes = [
    "Entity: Eiffel Tower | height: 330m | built: 1889 | designer: Gustave Eiffel",
]
sample = RAGSample(query=query, retrieved_docs=graph_nodes, answer=answer)
```

Future improvement: `context_type` field that adapts judge prompts for structured graph data vs prose chunks.

---

### Priority 2 — Statistical rigor

**Confidence intervals on batch results**

```python
# Current output
avg_score: 0.78

# With confidence intervals
avg_score: 0.78
ci_95: [0.71, 0.85]
n: 50
```

A drop from 0.80 to 0.78 with overlapping confidence intervals is noise. A drop from 0.80 to 0.65 with non-overlapping intervals is a real regression. This prevents false CI alarms and missed real failures.

---

### Priority 3 — Developer experience

**Regression Tracker**

Saves results to SQLite. `rageval history` shows pipeline quality over time.

```
Run History — faithfulness

v2.3  0.91  ▲ +0.06
v2.2  0.85  ▲ +0.03
v2.1  0.82  ▼ -0.04
v2.0  0.86
```

Teams cannot remove a tool that stores their history. This is the stickiness mechanism.

---

**zero-config quick_eval()**

```python
import rageval

# No judge setup. No metric config.
# Uses ANTHROPIC_API_KEY or OPENAI_API_KEY automatically.
result = rageval.quick_eval(
    query="...",
    docs=["..."],
    answer="..."
)
```

Reduces time from "heard about it" to "saw it work" from 15 minutes to 30 seconds.

---

**explain() method**

When a score is low, tells the user what to fix:

```python
explanation = rageval.explain(result)
# Output:
# Your faithfulness score is 0.43. This means 57% of claims are hallucinated.
# Most likely cause: your system prompt does not restrict the LLM to context only.
# Suggested fix: add "Answer ONLY using the provided context" to your system prompt.
```

---

**Web playground**

Browser interface at rageval.dev where anyone pastes query + docs + answer and gets scores back. No installation. No API key setup. Expands audience from Python developers to everyone building with LLMs.

---

## Metrics Coverage by RAG Type

| RAG Type | Faithfulness | ContextPrecision | AnswerRelevancy | ContextRecall | NoiseSensitivity | AnswerCompleteness | ContradictionDetector |
|---|---|---|---|---|---|---|---|
| Simple RAG | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ | ✅ |
| Advanced RAG (reranking) | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ | ✅ |
| Graph RAG | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ | ✅ |
| Agentic RAG | ✅ per step | ✅ per step | ✅ final | ✅ final | ✅ | ✅ | ✅ |
| Hybrid RAG | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ | ✅ |

\* requires ground_truth to be set in RAGSample

For all advanced types: stringify retrieved content before passing to RAGSample, or use `RetrievedDoc` for source tracking.

---

## File Reference

| File | Purpose | Key function/class |
|---|---|---|
| `rageval/core/sample.py` | Input contract | `RAGSample`, `retrieved_texts` |
| `rageval/core/result.py` | Output contract | `MetricResult`, `EvalResult` |
| `rageval/core/pipeline.py` | Orchestration | `evaluate()`, `batch_evaluate()`, `summary()` |
| `rageval/core/retrieved_doc.py` | Source-tracked doc | `RetrievedDoc` |
| `rageval/core/hallucination.py` | Hallucination types | `Hallucination`, `HallucinationType` |
| `rageval/metrics/base.py` | Metric contract | `BaseMetric`, `validate()`, `_make_result()` |
| `rageval/metrics/faithfulness.py` | Hallucination detection | `Faithfulness` |
| `rageval/metrics/context_precision.py` | Retrieval noise detection | `ContextPrecision` |
| `rageval/metrics/answer_relevancy.py` | Off-topic detection | `AnswerRelevancy` |
| `rageval/metrics/context_recall.py` | Retriever coverage | `ContextRecall` |
| `rageval/metrics/noise_sensitivity.py` | Pipeline robustness | `NoiseSensitivity` |
| `rageval/metrics/answer_completeness.py` | Answer coverage | `AnswerCompleteness` |
| `rageval/metrics/contradiction_detector.py` | Direct contradiction detection | `ContradictionDetector` |
| `rageval/judges/base.py` | LLM provider contract | `BaseJudge`, `complete_json()` |
| `rageval/judges/openai_judge.py` | OpenAI backend | `OpenAIJudge` |
| `rageval/judges/anthropic_judge.py` | Anthropic backend | `AnthropicJudge` |
| `rageval/judges/gemini_judge.py` | Google Gemini backend | `GeminiJudge` |
| `rageval/judges/ollama_judge.py` | Local Ollama backend | `OllamaJudge` |
| `rageval/judges/cohere_judge.py` | Cohere backend | `CohereJudge` |
| `rageval/judges/groq_judge.py` | Groq backend | `GroqJudge` |
| `rageval/judges/heuristic.py` | Free embedding backend | `HeuristicJudge`, `similarity()` |
| `rageval/tracker.py` | Regression tracking | `RunTracker` |
| `rageval/dataset.py` | Eval dataset generation | `EvalDatasetGenerator` |
| `rageval/reporters/json_csv.py` | Export | `to_json()`, `to_csv()` |
| `rageval/cli.py` | CLI | `run`, `history`, `diff`, `init`, `init-ci` |
| `examples/basic_eval.py` | Quickstart demo | — |

---

## Adding a New Metric (Agent Instructions)

To add a new metric, create one file in `rageval/metrics/`. Follow this exact pattern:

```python
# rageval/metrics/your_metric.py

from rageval.metrics.base import BaseMetric
from rageval.core.sample import RAGSample
from rageval.core.result import MetricResult

YOUR_PROMPT = """\
Your prompt here.
Use {placeholders} for dynamic content.
End with: Respond ONLY with a JSON object. No explanation. No markdown fences.
{{"score": 0.0, "reasoning": "...", "evidence": ["..."]}}
"""

class YourMetric(BaseMetric):
    name = "your_metric"
    required_inputs = ["query", "retrieved_docs", "answer"]  # adjust as needed

    def score(self, sample: RAGSample) -> MetricResult:
        self.validate(sample)  # always first

        try:
            result = self.judge.complete_json(
                YOUR_PROMPT.format(...)
            )
        except Exception as e:
            return self._make_result(
                score=0.0,
                reasoning=f"Metric failed: {str(e)}",
                evidence=[],
            )

        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "")
        evidence = result.get("evidence", [])

        return self._make_result(
            score=score,
            reasoning=reasoning,
            evidence=evidence,
        )
```

Then register it in `rageval/cli.py` metric_registry dict.

---

## Adding a New Judge (Agent Instructions)

To add a new LLM provider, create one file in `rageval/judges/`. Implement one method:

```python
# rageval/judges/your_judge.py

from rageval.judges.base import BaseJudge

class YourJudge(BaseJudge):

    def __init__(self, model: str = "default-model"):
        # initialize your SDK client here
        self.client = ...
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        # call your LLM API here
        # return the raw text response as a string
        response = self.client.call(prompt)
        return response.text
```

That is all. `complete_json()` is inherited from `BaseJudge` and handles JSON parsing automatically.

---

## Key Design Decisions

**Why evidence over score alone:** a score tells you there is a problem. Evidence tells you what the problem is. The entire value proposition of rageval is the evidence field in MetricResult.

**Why two prompts for faithfulness:** one prompt asking for both claim extraction and verification produces lower quality results. Two focused prompts — extract then verify — give the LLM one job at a time. This is chain-of-thought prompting applied at the architecture level.

**Why temperature=0 in all judges:** evaluation must be deterministic. Different scores on the same input make CI useless. Temperature 0 makes the LLM pick the highest probability token every time.

**Why try/except around every LLM call:** LLMs return malformed JSON, hit rate limits, and time out. One failed call should never destroy all other metric results. Errors are stored in MetricResult.reasoning so users can debug them.

**Why ThreadPoolExecutor for batch:** LLM calls are I/O-bound. Parallel threads give real speedup without async complexity. Keep max_workers at 4-10 to avoid rate limits.

**Why BaseJudge is an ABC:** the adapter pattern. Metrics talk to BaseJudge. Concrete judges wrap specific SDKs. Adding a new provider means one new file, zero changes to metrics.

---

## Current Limitations

1. No async native batch evaluation — ThreadPoolExecutor works but is not truly async
2. No confidence intervals on batch results — averages without statistical uncertainty
3. No step-by-step agentic evaluation — AgenticRAGSample not yet implemented
4. No Graph RAG native support — works via stringification but not optimized
5. No domain-specific embedding models — all-MiniLM-L6-v2 is general purpose only
6. No regression tracking — no history stored between runs
7. No web interface — CLI and Python only

---

## Version History

**v0.1.0**
- RAGSample and MetricResult data models
- Faithfulness metric with two-prompt chain-of-thought design
- ContextPrecision metric
- AnswerRelevancy metric with reverse generation
- evaluate() and batch_evaluate() pipeline
- OpenAI, Anthropic, and HeuristicJudge backends
- JSON and CSV reporters
- CLI with CI/CD exit codes

**v0.2.0**
- ContextRecall metric (requires ground_truth)
- NoiseSensitivity metric — unique to rageval, measures pipeline robustness under noise injection
- HallucinationClassifier — Faithfulness now returns structured Hallucination objects with type + severity
- RetrievedDoc with source tracking — retrieved_docs accepts list[str] or list[RetrievedDoc]
- GeminiJudge, OllamaJudge, CohereJudge, GroqJudge backends
- CLI supports: openai, anthropic, gemini, ollama, cohere, groq

**v0.3.0 — current**
- RunTracker — SQLite regression tracking, zero new dependencies
- EvalDatasetGenerator — generate eval datasets from raw documents
- AnswerCompleteness metric — catches what the answer leaves out
- ContradictionDetector metric — catches answers that reverse what context says
- CLI: `rageval history`, `rageval diff`, `rageval init`, `rageval init-ci`
- `--save-run` flag on `rageval run` to persist runs to tracker
- `RunTracker` and `EvalDatasetGenerator` exported from package root

**v0.4.0 — planned**
- AgenticRAGSample for multi-step pipelines
- PromptOptimizer — analyzes hallucination patterns and suggests prompt fixes
- Confidence intervals on batch results
- Regression tracker with alerting

**v1.0.0 — planned**
- Web playground
- arXiv paper with benchmark results
- Domain-specific embedding model support
- Full async batch evaluation