**For LangChain Discord (#showcase or #general):**
Hey everyone! If you are using LangChain for RAG and getting frustrated with evaluation tools that just spit out a float score (like `faithfulness: 0.43`) without telling you *why*, I just open-sourced a library called **rageval**. 

Instead of just a score, it returns the exact sentence that hallucinated and what the context actually said. It's completely framework-agnosticâ€”just pass it the `page_content` from your LangChain documents and the string output of your chain. It also supports local models via Ollama out of the box if you don't want to use OpenAI.

Repo: https://github.com/SujalS123/rageval
Would love any feedback from folks building production RAG here!

---

**For LlamaIndex Discord (#showcase or #general):**
Hey folks! I've been building RAG with LlamaIndex and got annoyed trying to debug my pipelines when eval scores dropped. So I built **rageval** â€” an evaluation library that returns the specific *evidence* that caused a score to drop, not just a number. 

If your model hallucinates, it tells you exactly which sentence was a hallucination. If precision drops, it tells you which retrieved node was irrelevant noise. 

You don't need any complex adapters to use it with LlamaIndex. Just extract `[n.text for n in response.source_nodes]` and pass it as plain strings. 

Docs: https://SujalS123.github.io/rageval
Hope this helps anyone else stuck debugging pipeline quality!

---

**For Hugging Face Discord / LocalLLaMA Reddit:**
Hey everyone, I just open-sourced **rageval** â€” a RAG evaluation library built with local models in mind.

A lot of RAG eval tools force you to use OpenAI or have really clunky integrations for local models. rageval supports `OllamaJudge` out of the box so you can run evaluations with local Llama 3 or Mistral entirely for free, privately on your own machine. It also includes a zero-cost `HeuristicJudge` for embedding-based similarity checks.

Most importantly: it returns the exact sentence that caused a metric to fail, not just a float score. 

Check it out: https://github.com/SujalS123/rageval (`pip install rageval[ollama]`)
Let me know if you run into any issues running it with your local setups!
