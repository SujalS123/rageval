from rageval.judges.base import BaseJudge

class OpenAIJudge(BaseJudge):
    """
    LLM "Judge backed by OpenAI models.
    
    Default model: gpt-4o-mini  - fast and chip for evaluation.
    Use gpt-4o for higher accuracy on complex evaluation 
    
    Reads OPENAI_API from enviroment automatically.
    set it with: export OpenAI_API_Key = YOur_key (Mac/Linux)
                 set OPENAI_API_KEY = your_key (Windows)
    """

    def __init__(self , model: str = "gpt-4o-mini" , api_key: str = None):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not found. "
                "Install it with: pip install openai"
            )
        self.client = openai.OpenAI(api_key = api_key)
        self.model  = model

    def complete(self , prompt : str , system: str = "") -> str:
        messages = []
        if system :
            messages.append({"role":"system" , "content":system})
        messages.append({"role": "user" ,"content": prompt})

        response = self.client.chat.completions.create(
            model = self.model ,
            messages = messages ,
            temperature = 0 ,# always 0 for evaluation - results must be deterministic
        )
        return response.choices[0].message.content
    