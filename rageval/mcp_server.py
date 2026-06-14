# rageval/mcp_server.py
"""
rageval MCP Server

Exposes rageval as a Model Context Protocol (MCP) server so AI coding
assistants (Claude Desktop, Cursor, Windsurf) can call it directly.

Install the optional dependency:
    pip install rageval[mcp]

Start the server:
    rageval serve
    # or directly:
    python -m rageval.mcp_server

Configure in Claude Desktop (~/.claude/claude_desktop_config.json):
    {
        "mcpServers": {
            "rageval": {
                "command": "rageval",
                "args": ["serve"]
            }
        }
    }

Configure in Cursor (settings.json):
    {
        "mcp.servers": {
            "rageval": {
                "command": "rageval serve"
            }
        }
    }

Configure in Windsurf:
    Add rageval to your MCP server list with command: rageval serve
"""

import json
from typing import Any


def create_server():
    """
    Create and return the MCP server instance.

    Requires: pip install mcp
    """
    try:
        from mcp.server import Server
        from mcp import types
    except ImportError:
        raise ImportError(
            "mcp package not found. Install it with: pip install rageval[mcp]"
        )

    server = Server("rageval")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="evaluate_rag",
                description=(
                    "Evaluate a RAG pipeline output for faithfulness, context precision, "
                    "and answer relevancy. Returns scores with evidence explaining failures."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The user query"},
                        "retrieved_docs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The documents retrieved by the RAG pipeline",
                        },
                        "answer": {"type": "string", "description": "The generated answer"},
                        "judge_type": {
                            "type": "string",
                            "enum": ["openai", "anthropic", "heuristic"],
                            "default": "openai",
                            "description": "Which LLM judge to use",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["faithfulness", "context_precision"],
                            "description": "Metrics to run",
                        },
                        "ground_truth": {
                            "type": "string",
                            "description": "Optional ground truth answer (required for context_recall)",
                        },
                    },
                    "required": ["query", "retrieved_docs", "answer"],
                },
            ),
            types.Tool(
                name="analyze_hallucination",
                description=(
                    "Check whether a specific claim is supported, unsupported, or "
                    "contradicted by the given context. Returns classification and reason."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "The claim to verify"},
                        "context": {"type": "string", "description": "The reference context"},
                        "judge_type": {
                            "type": "string",
                            "enum": ["openai", "anthropic"],
                            "default": "openai",
                        },
                    },
                    "required": ["claim", "context"],
                },
            ),
            types.Tool(
                name="classify_query",
                description=(
                    "Classify a query into one of: factual, comparison, multi_hop, "
                    "time_sensitive, negation, procedural, ambiguous, unanswerable."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "judge_type": {
                            "type": "string",
                            "enum": ["openai", "anthropic"],
                            "default": "openai",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="generate_eval_data",
                description=(
                    "Generate evaluation questions with ground truth answers from documents. "
                    "Output is directly usable by rageval run."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source documents to generate questions from",
                        },
                        "n_questions": {
                            "type": "integer",
                            "default": 10,
                            "description": "Number of questions to generate",
                        },
                        "judge_type": {
                            "type": "string",
                            "enum": ["openai", "anthropic"],
                            "default": "openai",
                        },
                    },
                    "required": ["documents"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        try:
            result = await _dispatch(name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


async def _dispatch(tool_name: str, args: dict) -> dict:
    if tool_name == "evaluate_rag":
        return _evaluate_rag(args)
    if tool_name == "analyze_hallucination":
        return _analyze_hallucination(args)
    if tool_name == "classify_query":
        return _classify_query(args)
    if tool_name == "generate_eval_data":
        return _generate_eval_data(args)
    raise ValueError(f"Unknown tool: {tool_name}")


def _build_judge(judge_type: str):
    if judge_type == "anthropic":
        from rageval.judges.anthropic_judge import AnthropicJudge
        return AnthropicJudge()
    from rageval.judges.openai_judge import OpenAIJudge
    return OpenAIJudge()


def _evaluate_rag(args: dict) -> dict:
    from rageval.core.sample import RAGSample
    from rageval.core.pipeline import evaluate
    from rageval.metrics.faithfulness import Faithfulness
    from rageval.metrics.context_precision import ContextPrecision
    from rageval.metrics.answer_relevancy import AnswerRelevancy
    from rageval.metrics.context_recall import ContextRecall

    judge = _build_judge(args.get("judge_type", "openai"))
    sample = RAGSample(
        query=args["query"],
        retrieved_docs=args["retrieved_docs"],
        answer=args["answer"],
        ground_truth=args.get("ground_truth"),
    )

    metric_map = {
        "faithfulness": Faithfulness(judge, threshold=0.8),
        "context_precision": ContextPrecision(judge, threshold=0.7),
        "answer_relevancy": AnswerRelevancy(judge, threshold=0.7),
        "context_recall": ContextRecall(judge, threshold=0.8),
    }
    selected = [metric_map[m] for m in args.get("metrics", ["faithfulness", "context_precision"]) if m in metric_map]
    result = evaluate(sample=sample, metrics=selected)
    return result.to_dict()


def _analyze_hallucination(args: dict) -> dict:
    judge = _build_judge(args.get("judge_type", "openai"))
    prompt = (
        f"Context:\n{args['context']}\n\n"
        f"Claim: {args['claim']}\n\n"
        "Is this claim supported, unsupported, or contradicted by the context? "
        "Respond ONLY with JSON: "
        '{"verdict": "supported|unsupported|contradicted", "reason": "..."}'
    )
    result = judge.complete_json(prompt)
    return {"claim": args["claim"], **result}


def _classify_query(args: dict) -> dict:
    judge = _build_judge(args.get("judge_type", "openai"))
    from rageval.query_classifier import QueryClassifier
    classifier = QueryClassifier(judge=judge)
    q_type = classifier.classify(args["query"])
    return {"query": args["query"], "query_type": q_type.value}


def _generate_eval_data(args: dict) -> dict:
    judge = _build_judge(args.get("judge_type", "openai"))
    from rageval.dataset import EvalDatasetGenerator
    gen = EvalDatasetGenerator(judge=judge)
    questions = gen.generate(
        documents=args["documents"],
        n_questions=args.get("n_questions", 10),
    )
    return {"questions": questions, "count": len(questions)}


if __name__ == "__main__":
    import asyncio
    try:
        from mcp.server.stdio import stdio_server
    except ImportError:
        raise ImportError("Install with: pip install rageval[mcp]")

    server = create_server()

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())
