# rageval/judges/gemini_judge.py

from rageval.judges.base import BaseJudge


class GeminiJudge(BaseJudge):
    """
    LLM judge backed by Google Gemini models.

    Default model: gemini-1.5-flash — fast and free-tier friendly.
    Use gemini-1.5-pro for higher accuracy.

    Reads GOOGLE_API_KEY from environment automatically.
    Set it with: export GOOGLE_API_KEY=your_key  (Mac/Linux)
                 set GOOGLE_API_KEY=your_key      (Windows)

    Note: Gemini's generate_content does not accept a separate system role
    in the same way as OpenAI. The system prompt is prepended to the user
    prompt with a clear separator so the model still follows it.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. "
                "Install it with: pip install google-generativeai"
            )
        if api_key:
            genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        model = self.genai.GenerativeModel(
            self.model,
            generation_config={"temperature": 0},
        )
        response = model.generate_content(full_prompt)
        return response.text
