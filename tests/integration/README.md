# Integration Tests

These tests make real LLM API calls. They validate that rageval works
correctly with actual LLM responses, not mocks.

## Setup

Set at least one API key:

    export ANTHROPIC_API_KEY=your_key_here
    export OPENAI_API_KEY=your_key_here
    export GROQ_API_KEY=your_key_here

## Run all integration tests

    pytest tests/integration/ -v -s

## Run with specific judge

    RAGEVAL_TEST_JUDGE=anthropic pytest tests/integration/ -v -s
    RAGEVAL_TEST_JUDGE=openai pytest tests/integration/ -v -s
    RAGEVAL_TEST_JUDGE=groq pytest tests/integration/ -v -s

## Run with specific model

    RAGEVAL_TEST_MODEL=claude-haiku-4-5 pytest tests/integration/ -v -s
    RAGEVAL_TEST_MODEL=gpt-4o-mini pytest tests/integration/ -v -s
    RAGEVAL_TEST_MODEL=llama-3.3-70b-versatile pytest tests/integration/ -v -s
    RAGEVAL_TEST_MODEL=llama-3.1-8b-instant pytest tests/integration/ -v -s

## Run one file at a time

    pytest tests/integration/test_faithfulness_real.py -v -s
    pytest tests/integration/test_end_to_end_real.py -v -s

## Save output to file

    pytest tests/integration/ -v -s 2>&1 | tee integration_results.txt

## What these tests validate

- Prompts produce valid JSON from real LLM responses
- Faithfulness catches real hallucinations
- ContextPrecision flags real irrelevant documents
- AnswerRelevancy scores off-topic answers lower than on-topic ones
- batch_evaluate preserves result order
- JSON and CSV export produce valid files
- Edge cases do not raise uncaught exceptions

## Expected cost

Using claude-haiku-4-5 or gpt-4o-mini:
- Full suite: approximately 50 to 80 LLM calls
- Estimated cost: under 0.10 USD

## If a test fails

1. Run with -s to see all print output
2. Add print(judge.complete(prompt)) temporarily to see raw LLM response
3. If JSON parsing failed you will see ValueError in the traceback
4. Fix the prompt or parsing logic in the metric file not the test
5. The tests reflect correct expected behavior and should not be changed
