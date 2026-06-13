# rageval/judges/groq_judge.py

from rageval.judges.base import BaseJudge


class GroqJudge(BaseJudge):
    """
    LLM judge backed by Groq's inference API.

    Default model: llama-3.1-8b-instant — free tier friendly and extremely fast.
    Use llama-3.1-70b-versatile for higher accuracy.

    Reads GROQ_API_KEY from environment automatically.
    Set it with: export GROQ_API_KEY=your_key  (Mac/Linux)
                 set GROQ_API_KEY=your_key      (Windows)

    Groq runs open-source models on custom LPU hardware — latency is
    typically 10-50x faster than OpenAI at a fraction of the cost.
    Recommended for development iteration and CI pipelines.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None):
        try:
            import groq
        except ImportError:
            raise ImportError(
                "groq package not found. "
                "Install it with: pip install groq"
            )
        self.client = groq.Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
