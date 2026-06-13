# Gemini Judge

```python
from rageval.judges.gemini_judge import GeminiJudge

judge = GeminiJudge()                              # reads GEMINI_API_KEY
judge = GeminiJudge(model="gemini-1.5-flash")      # default
judge = GeminiJudge(model="gemini-1.5-pro")        # higher accuracy
```

## Install

```bash
pip install rageval[gemini]
```

## Set API key

```bash
export GEMINI_API_KEY=your_key_here
```
