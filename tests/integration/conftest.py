# tests/integration/conftest.py
"""
Integration test configuration.
All tests here make real LLM API calls.

Run with:
    pytest tests/integration/ -v --tb=short

Required environment variables (at least one judge must be set):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY

Optional:
    RAGEVAL_TEST_JUDGE = "anthropic" or "openai" (default: auto-detect)
    RAGEVAL_TEST_MODEL = model name override
"""

import os
import pytest
from rageval.judges.anthropic_judge import AnthropicJudge
from rageval.judges.openai_judge import OpenAIJudge
from rageval.judges.groq_judge import GroqJudge
from rageval.judges.heuristic import HeuristicJudge


def get_judge():
    """
    Return the best available judge based on environment variables.
    Skips test if no API key is found.
    """
    preferred = os.environ.get("RAGEVAL_TEST_JUDGE", "auto").lower()

    if preferred == "anthropic" or (preferred == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        model = os.environ.get("RAGEVAL_TEST_MODEL", "claude-haiku-4-5")
        return AnthropicJudge(model=model)

    if preferred == "openai" or (preferred == "auto" and os.environ.get("OPENAI_API_KEY")):
        model = os.environ.get("RAGEVAL_TEST_MODEL", "gpt-4o-mini")
        return OpenAIJudge(model=model)

    if preferred == "groq" or (preferred == "auto" and os.environ.get("GROQ_API_KEY")):
        model = os.environ.get("RAGEVAL_TEST_MODEL", "llama-3.3-70b-versatile")
        return GroqJudge(model=model)

    pytest.skip("No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY to run integration tests.")


@pytest.fixture(scope="session")
def judge():
    return get_judge()


@pytest.fixture(scope="session")
def embedding_judge():
    return HeuristicJudge()


# ── Shared test samples ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def perfect_sample():
    """Answer perfectly supported by context. All metrics should score high."""
    from rageval.core.sample import RAGSample
    return RAGSample(
        query="What is the boiling point of water at sea level?",
        retrieved_docs=[
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
            "The boiling point of water decreases at higher altitudes due to lower atmospheric pressure.",
        ],
        answer="Water boils at 100 degrees Celsius at sea level.",
    )


@pytest.fixture(scope="session")
def hallucination_sample():
    """Answer contains a clear hallucination. Faithfulness should score low."""
    from rageval.core.sample import RAGSample
    return RAGSample(
        query="Who designed the Eiffel Tower?",
        retrieved_docs=[
            "The Eiffel Tower was designed by engineer Gustave Eiffel and constructed between 1887 and 1889.",
            "The tower stands 330 meters tall and was the tallest man-made structure for 41 years.",
        ],
        answer="The Eiffel Tower was designed by Leonardo da Vinci in 1887 and stands 330 meters tall.",
    )


@pytest.fixture(scope="session")
def noisy_retrieval_sample():
    """Retrieved docs contain irrelevant noise. ContextPrecision should score low."""
    from rageval.core.sample import RAGSample
    return RAGSample(
        query="What is Python used for?",
        retrieved_docs=[
            "Python is a high-level programming language used for web development, data science, and automation.",
            "The Amazon rainforest covers 5.5 million square kilometers in South America.",
            "Python supports multiple programming paradigms including procedural and object-oriented.",
            "French cuisine is known for its use of butter, wine, and fresh ingredients.",
        ],
        answer="Python is used for web development, data science, and automation.",
    )


@pytest.fixture(scope="session")
def off_topic_answer_sample():
    """Answer does not address the query. AnswerRelevancy should score low."""
    from rageval.core.sample import RAGSample
    return RAGSample(
        query="What is the capital of France?",
        retrieved_docs=[
            "France is a country in Western Europe. Its capital city is Paris.",
            "Paris has a population of approximately 2 million in the city proper.",
        ],
        answer="France has a rich culinary tradition including baguettes, croissants, and fine wines. "
               "French cuisine is recognized as a UNESCO cultural heritage.",
    )


@pytest.fixture(scope="session")
def recall_sample():
    """Sample with ground truth for testing ContextRecall."""
    from rageval.core.sample import RAGSample
    return RAGSample(
        query="Tell me about the Eiffel Tower.",
        retrieved_docs=[
            "The Eiffel Tower was constructed between 1887 and 1889.",
            "It was designed by Gustave Eiffel.",
        ],
        answer="The Eiffel Tower was built between 1887 and 1889 by Gustave Eiffel.",
        ground_truth=(
            "The Eiffel Tower was constructed between 1887 and 1889. "
            "It was designed by engineer Gustave Eiffel. "
            "The tower stands 330 meters tall. "
            "It was the tallest man-made structure for 41 years."
        ),
    )
