# Changelog

## v0.5.0 — current

### Added
- ConsistencyAnalyzer — measures answer consistency across query paraphrases
- FailureTaxonomyBuilder — clusters failures by root cause automatically
- PromptVersionControl — tracks prompt versions and compares their impact
- SemanticDriftDetector — detects when production queries drift from knowledge base
- ExplainabilityReporter — generates HTML reports with sentence-level annotation

## v0.4.0

### Added
- RAGTracer — step-level pipeline tracing with root cause identification
- QueryClassifier — classifies queries into 8 types, shows per-type performance
- ChunkQualityAnalyzer — analyzes chunk quality before indexing, zero LLM calls
- AutoEval — production monitoring decorator with background evaluation
- MCP Server — rageval as an MCP tool for AI coding assistants

## v0.3.0

### Added
- RunTracker — SQLite-based regression tracking, rageval history and diff commands
- EvalDatasetGenerator — generates eval datasets from documents automatically
- AnswerCompleteness metric
- ContradictionDetector metric
- rageval init-ci — generates GitHub Actions workflow
- rageval init — creates starter eval dataset

## v0.2.0

### Added
- ContextRecall metric (requires ground_truth)
- NoiseSensitivity metric — unique to rageval
- HallucinationClassifier — Faithfulness returns structured Hallucination objects
- RetrievedDoc with source tracking
- GeminiJudge, OllamaJudge, CohereJudge, GroqJudge

## v0.1.0 — initial release

### Added
- RAGSample, MetricResult, EvalResult data models
- Faithfulness, ContextPrecision, AnswerRelevancy metrics
- OpenAIJudge, AnthropicJudge, HeuristicJudge
- evaluate(), batch_evaluate(), summary()
- JSON and CSV reporters
- CLI with CI/CD exit codes
