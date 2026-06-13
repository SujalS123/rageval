#rageval/judges/base.py

from abc import ABC , abstractmethod
import json
import re

class BaseJudge(ABC):
    """
    Abstract base class for all LLm judge backends.
    
    Every judge implements one method: complete().
    Everything else in the ibrary calls complete() or complete_json().
    swapping from OpenAI to Claude to OLLama = changing one line for the user.
    
    This is the Adapter design pattern:
    -Your metrics are the "client" - they only know about BaseJudge
    -OpenAIJudge , Anthropic etc are the "adapters" - they wrap specific SDKs
    -Adding a new LLM provider = writing one new adapter class

    """

    @abstractmethod
    def complete(self , prompt: str , system: str = "")-> str:
        """
        Send a prompt to the LLM and return the raw text response.
        
        Args:
            prompt: the user message to send
            system: optional system message (instructions for the LLM)
            
        Returns:
            Raw text response from the LLM
        """
        raise NotImplementedError
    
    def complete_json(self , prompt: str , system: str = "") -> dict:
        """
        Send a prompt , expect a JSON response , parse and return it as a dict.

        Handles the most commmon LLm formattinng issues:
        -JSON wrapped in markdown code fences (```json ...```)
        -Extra whitespace around the JSON
        -JSON somwhere inside a longer response

        why this method exists: LLMs ofthen add explanation text arounf JSON 
        even when you tell them not to . this defensive parsing handles that.
        """
        raw = self.complete(prompt , system)

        #Attempt 1 : direct parse(cleanest case)
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        #Attempt 2: strip markdowm code fences and retry 
        #Handles: ```json\n{...}\n```and ```\n{...}n```
        cleaned = re.sub(r'```(?:json)?\n?|\n?```', '', raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        #Attempt 3: find first {...} block in the response
        match = re.search(r'\{.*\}',raw , re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # All attempts failed - raise with the raw output so user can debug 
        raise ValueError(
            f"Judge returned a response that could not be parsed as JSON. \n"
            f"Raw response was: \n{raw}\n\n"
            f"Check your prompt - it should say 'Return ONLY valid JSON'."
        )
    
    
"""

***Understand the three-attempt JSON parsing:**

Attempt 1 handles the idal case - the LLM returned clean JSON.

Attempt 2 handles the most common failure - the LLM wrapped JSON in markdown fences like this: 
```
```json
{"score": 0.8}

"""
        


