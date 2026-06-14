# Ollama Judge

Runs any local model through Ollama. No API key. No data leaves your machine.

```python
from rageval.judges.ollama_judge import OllamaJudge

judge = OllamaJudge()                      # defaults to llama3, localhost:11434
judge = OllamaJudge(model="mistral")
judge = OllamaJudge(model="llama3", base_url="http://localhost:11434")
```

## Install

```bash
pip install rageval-core[ollama]
```

Also install Ollama from [ollama.ai](https://ollama.ai) and pull a model:

```bash
ollama pull llama3
```

## Use case

Air-gapped environments. Private data that cannot leave your infrastructure. Development without API costs.
