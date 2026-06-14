import asyncio
from rageval import aevaluate, RAGSample
from rageval.metrics.faithfulness import Faithfulness
from rageval.judges.heuristic import HeuristicJudge

async def main():
    print("Initializing judge and metric...")
    judge = HeuristicJudge()
    metric = Faithfulness(judge=judge)
    
    sample = RAGSample(
        query="What is testing?",
        retrieved_docs=["Testing is a process to verify code works."],
        answer="Testing verifies code works."
    )
    
    print("Running aevaluate...")
    result = await aevaluate(sample, metrics=[metric])
    
    print("Result summary:")
    print(result.summary())
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
