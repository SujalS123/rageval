# Groq Judge

Groq runs open-source models on custom LPU hardware. Latency is typically 10–50x faster than OpenAI. The free tier is sufficient for development and CI pipelines.

```python
from rageval.judges.groq_judge import GroqJudge

judge = GroqJudge()                                        # reads GROQ_API_KEY
judge = GroqJudge(model="llama-3.3-70b-versatile")         # best quality
judge = GroqJudge(model="llama-3.1-8b-instant")            # default, fastest
```

## Install

```bash
pip install rageval[groq]
```

## Set API key

```bash
export GROQ_API_KEY=your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Models

- `llama-3.1-8b-instant` — default, extremely fast
- `llama-3.3-70b-versatile` — higher accuracy, still faster than most cloud APIs
