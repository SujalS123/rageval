# Installation

## Basic install

```bash
pip install rageval
```

The base install includes the HeuristicJudge (free, local embeddings) and all evaluation metrics. No API key required to start.

## With a specific judge

```bash
pip install rageval[anthropic]   # Anthropic Claude
pip install rageval[openai]      # OpenAI GPT-4o / GPT-4o-mini
pip install rageval[gemini]      # Google Gemini
pip install rageval[groq]        # Llama 3 on Groq (free tier)
pip install rageval[ollama]      # Any local Ollama model
pip install rageval[cohere]      # Cohere Command R
```

## Everything

```bash
pip install rageval[all]
```

## MCP server

```bash
pip install rageval[mcp]   # rageval as an MCP tool for AI coding assistants
```

## Requirements

Python 3.10 or higher.

## Verify the install

```python
import rageval
print(rageval.__version__)  # 0.5.0
```
