# FAQ

**Does rageval store my data?**
No. All evaluation runs locally. LLM calls go directly from your machine to your chosen provider. `rageval` stores nothing remotely.

**How much does it cost to run?**
Using `claude-3-5-haiku` via Anthropic: approximately $0.001 per sample for faithfulness + context_precision + answer_relevancy combined. 100 samples cost roughly $0.10. `HeuristicJudge` is completely free.

**Can I use rageval without an API key?**
Yes. `HeuristicJudge` uses local sentence embeddings and costs nothing. It is less accurate than LLM judges but great for rapid development and testing in CI/CD without burning credits.

**Does rageval work with LangChain or LlamaIndex?**
Yes. Pass plain strings — extract `query`, `retrieved_docs`, and `answer` from whatever framework you use. No framework-specific objects or complex adapters needed.

**How is this different from other tools?**
Other evaluation frameworks return a float score. `rageval` returns a diagnosis: the specific claims that failed, classified by hallucination type with severity scores. It's built for debugging, not just measuring.

**What Python versions are supported?**
Python 3.10, 3.11, and 3.12. Tested extensively on all three in our CI pipeline.

**Can I add custom metrics?**
Yes. Inherit `BaseMetric`, implement `score()`, and return a `MetricResult`.

**Is rageval production ready?**
Yes. The core metrics are validated with extensive integration tests against real LLM calls. Our advanced modules (`RunTracker`, `ConsistencyAnalyzer`) are designed specifically for production stability and CI/CD pipelines.
