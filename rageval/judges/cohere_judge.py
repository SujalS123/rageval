# rageval/judges/cohere_judge.py

from rageval.judges.base import BaseJudge


class CohereJudge(BaseJudge):
    """
    LLM judge backed by Cohere models.

    Default model: command-r — balanced speed and quality.
    Use command-r-plus for higher accuracy.

    Reads COHERE_API_KEY from environment automatically.
    Set it with: export COHERE_API_KEY=your_key  (Mac/Linux)
                 set COHERE_API_KEY=your_key      (Windows)
    """

    def __init__(self, model: str = "command-r", api_key: str = None):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package not found. "
                "Install it with: pip install cohere"
            )
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.message.content[0].text
