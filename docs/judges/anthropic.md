# Anthropic Judge

```python
from rageval.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge()                            # reads ANTHROPIC_API_KEY
judge = AnthropicJudge(model="claude-haiku-4-5")    # default — fast and cheap
judge = AnthropicJudge(model="claude-sonnet-4-5")   # higher accuracy
judge = AnthropicJudge(api_key="sk-ant-...")        # explicit key
```

## Install

```bash
pip install rageval[anthropic]
```

## Set API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Models

- `claude-haiku-4-5` — default, recommended (fastest, cheapest Claude)
- `claude-sonnet-4-5` — higher accuracy
