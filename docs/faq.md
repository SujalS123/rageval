# FAQ

**Does rageval store my data?**
No. All evaluation runs locally. LLM calls go directly from your
machine to your chosen provider. rageval stores nothing remotely.

**How much does it cost to run?**
Using claude-haiku-4-5: approximately $0.001 per sample for
faithfulness + context_precision + answer_relevancy combined.
100 samples costs roughly $0.10. HeuristicJudge is completely free.

**Can I use rageval without an API key?**
Yes. HeuristicJudge uses local sentence embeddings and costs nothing.
It is less accurate than LLM judges but good for development.

**Does rageval work with LangChain / LlamaIndex?**
Yes. Pass plain strings — extract query, retrieved_docs, and answer
from whatever framework you use. No framework objects needed.

**How is this different from RAGAS?**
RAGAS returns a float score. rageval returns the specific claims
that failed, classified by hallucination type with severity scores.
rageval also works without LangChain and supports local LLM judges.

**What Python versions are supported?**
Python 3.10, 3.11, and 3.12. Tested on all three in CI.

**Can I add custom metrics?**
Yes. Inherit BaseMetric, implement score(), return a MetricResult.
See CONTRIBUTING.md and the Adding a New Metric section in GUIDE.md.

**Is rageval production ready?**
The core metrics (Faithfulness, ContextPrecision, AnswerRelevancy)
are validated with 41 integration tests against real LLM calls.
Advanced features are newer and should be evaluated for your use case.
