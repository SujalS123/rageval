# CI/CD Integration

rageval run exits with code 1 if scores fall below threshold. One line in GitHub Actions blocks deployments when RAG quality drops.

## GitHub Actions setup

```yaml
name: RAG evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install rageval-core[anthropic]
      - name: Evaluate RAG pipeline
        run: rageval run eval_data.json --judge anthropic --threshold 0.8
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## eval_data.json format

```json
[
  {
    "query": "What is Python?",
    "retrieved_docs": ["Python is a high-level programming language..."],
    "answer": "Python is a programming language created by Guido van Rossum."
  },
  {
    "query": "What is the boiling point of water?",
    "retrieved_docs": ["Water boils at 100 degrees Celsius at sea level."],
    "answer": "Water boils at 100 degrees Celsius.",
    "ground_truth": "Water boils at 100 degrees Celsius (212 Fahrenheit) at sea level."
  }
]
```

## Generate a starter file

```bash
rageval init
```

This creates `eval_data.json` with example samples. Edit it with your own data.

## Generate the GitHub Actions workflow

```bash
rageval init-ci --judge anthropic
```

This creates `.github/workflows/rag-eval.yml` with the correct API key secret configured.

## Available judge options for CLI

```bash
rageval run eval_data.json --judge anthropic --threshold 0.8
rageval run eval_data.json --judge openai --model gpt-4o-mini --threshold 0.8
rageval run eval_data.json --judge groq --model llama-3.3-70b-versatile --threshold 0.8
rageval run eval_data.json --judge ollama --model llama3 --threshold 0.8
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All metrics passed threshold |
| 1 | One or more metrics below threshold |

## Saving run history

```bash
rageval run eval_data.json --judge anthropic --threshold 0.8 --save-run v2.3-deploy
```

This saves the run to the local SQLite tracker. View history with:

```bash
rageval history
rageval diff v2.2-deploy v2.3-deploy
```
