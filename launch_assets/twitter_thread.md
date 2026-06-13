**Tweet 1:**
Every RAG evaluation tool tells you `faithfulness: 0.43`. 
None of them tell you which sentence hallucinated, which document was noise, or what to fix.

So I built one that does.

Introducing rageval: RAG evaluation that tells you why. 🧵👇
[Link to GitHub / Landing Page]
[Attach Image: The terminal output showing the Romeo & Juliet Dickens hallucination]

---

**Tweet 2:**
The problem with RAG evaluation today is that a float score doesn't help you debug. 
If precision is 0.5, is the retriever missing context? Is it fetching noise?

rageval extracts the exact claims from the LLM and cross-references them against your docs, returning the *evidence*.

---

**Tweet 3:**
It is completely framework agnostic. 
No LangChain chains required. No LlamaIndex query engines. 
Just pass plain strings. Extract your query, retrieved docs, and answer, and pass them to `RAGSample`. That's it.
[Attach Image: Code snippet showing RAGSample with raw strings]

---

**Tweet 4:**
You shouldn't be locked into OpenAI to evaluate your pipeline.
rageval supports 7 judges out of the box:
🧠 Anthropic
⚡ Groq (Llama 3)
🌐 Gemini
🇨🇦 Cohere
🦙 Ollama (100% local, privacy first)
🤖 OpenAI
[Attach Image: Code snippet showing AnthropicJudge and OllamaJudge]

---

**Tweet 5:**
It also ships with advanced tools for production RAG:
🔍 FailureTaxonomyBuilder: Clusters 1000 failing evaluations by root cause automatically.
🔄 ConsistencyAnalyzer: Catches when paraphrasing the same query breaks your pipeline.
📈 RunTracker: Built-in SQLite regression tracking.

---

**Tweet 6:**
It's fully open source (MIT). 
I built this because I wasted two weeks manually reading RAG outputs to debug a production failure. Never again.

Check it out on GitHub, drop a star if you find it useful, and let me know what you think! 
[Link to GitHub]
`pip install rageval`
