**Title:** Show HN: rageval – RAG evaluation that returns evidence, not just a score

**Post Body:**
Hey HN,

I built rageval because I was frustrated with existing RAG evaluation tools (like RAGAS or DeepEval) that return a float score (e.g., `faithfulness: 0.43`) and stop there. 

A score tells you something is wrong, but it doesn't tell you *what* is wrong. Did the LLM fabricate a specific claim? Did the retriever grab irrelevant noise? Without the evidence, you're stuck manually reading through hundreds of LLM outputs to debug your pipeline. 

rageval solves this by returning the specific evidence that caused the score. If faithfulness drops, it returns the exact sentence that hallucinated, classifies the type of hallucination (e.g., `FACTUAL_ERROR`), and lists what the context actually said.

A few other design decisions I made:
1. **Framework Agnostic:** It only takes plain strings. You don't have to pass LangChain or LlamaIndex objects into it.
2. **Pluggable Judges:** You aren't forced to use OpenAI. It supports Anthropic, Gemini, Cohere, Groq, and Ollama (for 100% local evaluation).
3. **Actionable Production Tools:** It includes a `FailureTaxonomyBuilder` that clusters batch evaluation failures by root cause, and a `ConsistencyAnalyzer` to detect paraphrase instability.

It's MIT licensed and available via `pip install rageval`. 

Repo: https://github.com/sujalsonawane/rageval
Docs: https://sujalsonawane.github.io/rageval/

I'd love to hear your feedback on the API design and what metrics/judges you'd like to see next.

**First Comment (To post immediately after):**
A quick example of what the terminal output looks like when it catches a hallucination (Query: "Who wrote Romeo and Juliet?", Answer: "Charles Dickens"):

```text
[FAIL] faithfulness : 0.00 (threshold=0.80)
Reasoning: 1 of 1 claims could not be verified from the context.
 Evidence:
 - FACTUAL_ERROR: 'Romeo and Juliet was written by Charles Dickens'
   (severity: 1.0) — Context states William Shakespeare
```
It gives you the diagnosis, not just the score. Happy to answer any questions about how the claim extraction works under the hood!
