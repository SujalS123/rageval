#rageval/judges/anthropic_judge.py

from rageval.judges.base import BaseJudge

class AnthropicJudge(BaseJudge):
    """
    LLM judge backed by Anthropic claude models.
    
    Default model: claude-haiku-4.5 - fastesta and cheapest claude model.
    use claude-sonnet-4.6 for higher accuracy.
    
    Reads Anthropic_API_KEY from enviroment automatically.
    set it with : export ANTHROPIC_API_KEY = your_key(Mac/Linux)
                  set ANTHROPIC_API_KEY = your_key (Windows)"""
    
    def __init__(self , model: str = "claude-haiku-4-5" , api_key : str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not found. "
                "Install itwith: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key = api_key)
        self.model = model

    def complete(self , prompt: str , system: str = "") -> str:
        kwargs ={
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user" , "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text