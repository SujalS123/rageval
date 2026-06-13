# OpenAI Judge

```python
from rageval.judges.openai_judge import OpenAIJudge

judge = OpenAIJudge()                        # reads OPENAI_API_KEY from environment
judge = OpenAIJudge(model="gpt-4o")          # higher accuracy
judge = OpenAIJudge(model="gpt-4o-mini")     # default — fast and cheap
judge = OpenAIJudge(api_key="sk-...")        # explicit key
```

## Install

```bash
pip install rageval[openai]
```

## Set API key

```bash
export OPENAI_API_KEY=your_key_here
```

## Models

- `gpt-4o-mini` — default, recommended for evaluation (fast, cheap, good JSON output)
- `gpt-4o` — higher accuracy for complex evaluations
