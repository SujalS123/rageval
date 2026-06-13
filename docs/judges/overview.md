# Judges Overview

A judge is the LLM that rageval uses to evaluate your RAG pipeline. Every judge implements one method: `complete()`. rageval metrics never import OpenAI or Anthropic directly — they only talk to `BaseJudge`. Swapping providers means changing one line.

## The adapter pattern

```python
from rageval.judges.base import BaseJudge

class BaseJudge(ABC):
    def complete(self, prompt: str, system: str = "") -> str:
        ...  # implemented by each provider

    def complete_json(self, prompt: str, system: str = "") -> dict:
        ...  # inherited — handles JSON parsing with 3-attempt fallback
```

Metrics call `judge.complete_json()`. The base class handles JSON extraction, markdown fence stripping, and retry logic. Concrete judges implement only `complete()`.

## Available judges

| Judge | Provider | Cost | Install |
|---|---|---|---|
| AnthropicJudge | Claude | Per token | built-in |
| OpenAIJudge | GPT-4o / GPT-4o-mini | Per token | built-in |
| GeminiJudge | Gemini | Per token | pip install rageval[gemini] |
| GroqJudge | Llama 3 on Groq | Free tier | pip install rageval[groq] |
| OllamaJudge | Any local model | Free | pip install rageval[ollama] |
| CohereJudge | Command R | Per token | pip install rageval[cohere] |
| HeuristicJudge | Local embeddings | Free | built-in |

## Writing a custom judge

Create one file, implement one method:

```python
# rageval/judges/your_judge.py
from rageval.judges.base import BaseJudge

class YourJudge(BaseJudge):

    def __init__(self, model: str = "your-default-model"):
        self.client = YourSDK()
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        response = self.client.call(model=self.model, prompt=prompt)
        return response.text
```

That is all. `complete_json()` is inherited and handles JSON parsing automatically. All metrics will work with your judge immediately.

## Choosing a judge

**For production evaluation:** AnthropicJudge with claude-haiku-4-5 or OpenAIJudge with gpt-4o-mini. Fast and cheap.

**For development iteration:** GroqJudge (free tier, very fast) or HeuristicJudge (no API calls at all).

**For air-gapped environments:** OllamaJudge with any local model.

**For AnswerRelevancy similarity computation:** HeuristicJudge. It handles embedding-based cosine similarity — no API calls needed for this step.
