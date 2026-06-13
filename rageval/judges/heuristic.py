# ragas/judges/heuristic.py
import numpy as np

class HeuristicJudge:
    """
    Embedding-based judge. no LLM calls. completely free to run
    
    Use this during Devlopment to iterate quickly without burning API tokens.
    Switch to OpenAIJudge or AnthropicJudge for production evalution runs.
    
    Does NOT inherit BaseJudge because it cannot do free-form completion.
    It only does similarity scoring using sentence embeddings.
    
    How cosine similarity works:
    - Sentences are converted to vectors (lists of 384 numbers)
    - Similar sentences produce vectors pointing in similar directions
    -cosine_similarity measures the angle between vectors
    -1.0 = identical meaning ,0.0 = completely unrelated
    """

    def __init__(self , model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformer package not found. "
                "Install it with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def similarity(self , text_a : str , text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        Returns a float between 0.0 and 1.0
        """
        embeddings = self.model.encode([text_a ,text_b])
        a , b = embeddings[0] , embeddings[1]

        dot_product = np.dot(a,b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)

        cosine_sim = dot_product / (magnitude_a * magnitude_b)

        # Clamp to [0,1] - cosine similarity can technically be slightly
        #negative for unrelated sentences , but for our purposes 0.0 is the floor
        return float(max(0.0 , min(1.0 , cosine_sim)))
    
    def batch_similarity(self , anchor: str , texts: list[str]) -> list[float]:
        """
        Compute similarity between one anchor text and list of texts.
        More Efficient than calling similarity() in a loop because
        sentence-transformerscan batch encode multiple texts at once.
        
        used by AnswerRelevancy to compare original query against
        multiple reverse-genrated questions at once.
        """
        if not texts:
            return []
        
        all_texts = [anchor] + texts
        embeddings = self.model.encode(all_texts)

        anchor_emb = embeddings[0]
        results = []

        for emb in embeddings[1:]:
            dot = np.dot(anchor_emb, emb)   
            mag_a = np.linalg.norm(anchor_emb)
            mag_b = np.linalg.norm(emb)
            sim = dot / (mag_a * mag_b)
            results.append(float(max(0.0 , min(1.0 , sim))))

        return results