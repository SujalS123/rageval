# rageval/judges/ollama_judge.py

from rageval.judges.base import BaseJudge


class OllamaJudge(BaseJudge):
    """
    LLM judge backed by a locally running Ollama instance.

    Default model: llama3. Any model pulled via `ollama pull <name>` works.

    No API key needed. Requires Ollama running at http://localhost:11434.
    Start it with: ollama serve

    This enables fully local evaluation with zero external API calls —
    ideal for air-gapped environments or cost-sensitive development.
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx package not found. "
                "Install it with: pip install httpx"
            )
        self.httpx = httpx
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, system: str = "") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if system:
            payload["system"] = system

        response = self.httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"]
