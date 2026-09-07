"""Command-line entry point for producing deterministic dataset profiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .embeddings import DEFAULT_MODEL_ID
from .metrics import evaluate_candidates, evaluate_funnel, evaluate_relationships
from .evaluation import candidate_recall_at_k, load_ground_truth
from .openai_provider import OpenAIConfig, OpenAIReasoningClient
from .matching import generate_candidates
from .ontology import build_canonical_schema
from .batch_reasoning import DEFAULT_REASONING_BATCH_SIZE, batch_reasoning_plan, reason_candidate_batches
from .cache import FileCache
from .pipeline_cache import PipelineCache
from .routing import DEFAULT_AMBIGUITY_MARGIN, DEFAULT_ROUTING_MINIMUM_SCORE
from .validation import finalize_mappings

from .models import DatasetRef, MatchMethod

def _diagnostics_output(diagnostics: object, include_pair_evidence: bool) -> dict[str, object]:
    output = asdict(diagnostics)
    if not include_pair_evidence:
        output.pop("pair_evidence", None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile local CSV datasets for schema mapping.")
    parser.add_argument("datasets", nargs="+", help="One or more CSV paths")
    parser.add_argument("--output", type=Path, help="Optional path for JSON profile output")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache", action="store_true", help="Reuse local cached pipeline and reasoning artifacts")
    cache_group.add_argument("--no-cache", action="store_true", help="Explicitly bypass the local artifact cache")
    parser.add_argument("--sample-size", type=int, default=5, help="Maximum distinct samples per column")
    parser.add_argument("--match", action="store_true", help="Generate lexical candidates for exactly two datasets")
    parser.add_argument("--embedding-match", action="store_true", help="Generate ONNX embedding candidates for exactly two datasets")
    parser.add_argument("--reason", action="store_true", help="Classify embedding candidates with the configured OpenAI model")
    parser.add_argument("--validate", action="store_true", help="Finalize reasoning with deterministic validation")
    parser.add_argument("--canonical", action="store_true", help="Generate a canonical schema from finalized mappings")
    parser.add_argument("--routing-minimum-score", type=float, default=DEFAULT_ROUTING_MINIMUM_SCORE, help="Minimum top embedding score required for LLM routing")
    parser.add_argument("--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN, help="Top-to-runner-up score gap that routes both candidates")
    parser.add_argument("--pair-evidence", action="store_true", help="Include every retained and excluded pair diagnostic in embedding JSON output")
    parser.add_argument("--reasoning-batch-size", type=int, default=DEFAULT_REASONING_BATCH_SIZE, help="Maximum routed candidates per source-column OpenAI request")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_ID, help="Hugging Face ONNX model identifier")
    parser.add_argument("--top-k", type=int, default=3, help="Candidates retained per source column")
    parser.add_argument("--minimum-score", type=float, help="Optional candidate score threshold")
    parser.add_argument("--ground-truth", type=Path, help="Optional mapping fixture for candidate recall@K")
    parser.add_argument("--evaluate", action="store_true", help="Emit fixture-limited metrics for the selected workflow")
    parser.add_argument("--benchmark", action="store_true", help="Compare normalized exact, lexical, and embedding retrieval")
    args = parser.parse_args()

    if (args.evaluate or args.benchmark) and not args.ground_truth:
        parser.error("--evaluate and --benchmark require --ground-truth")
    if args.benchmark and (args.match or args.embedding_match or args.reason or args.validate or args.canonical):
        parser.error("--benchmark runs retrieval strategies itself; do not combine it with workflow flags")
    if args.reason and not args.embedding_match:
        parser.error("--reason requires --embedding-match")
    if args.canonical and not args.validate:
        parser.error("--canonical requires --validate, --reason, and --embedding-match")
    if args.validate and not args.reason:
        parser.error("--validate requires --reason and --embedding-match")
    if args.match and args.embedding_match:
        parser.error("choose either --match or --embedding-match")
    cache_pipeline = PipelineCache(FileCache() if args.cache else None)
    profiles = [cache_pipeline.profile(DatasetRef.from_path(path), args.sample_size) for path in args.datasets]
    output = {"datasets": [profile.to_dict() for profile in profiles]}
    if args.benchmark:
        if len(profiles) != 2:
            parser.error("--benchmark requires exactly two CSV datasets")
        ground_truth = load_ground_truth(args.ground_truth)
        lexical_candidates = generate_candidates(profiles[0], profiles[1], top_k=args.top_k, minimum_score=0.6)
        embedding_retrieval = cache_pipeline.embedding_retrieval(
            profiles[0], profiles[1], model_id=args.embedding_model,
            top_k=args.top_k, minimum_score=0.35,
        )
        embedding_candidates = embedding_retrieval.candidates
        routing = cache_pipeline.routing(
            embedding_candidates, embedding_retrieval.diagnostics,
            minimum_score=args.routing_minimum_score, ambiguity_margin=args.ambiguity_margin,
        )
        batch_plan = batch_reasoning_plan(routing.reasoning_candidates, args.reasoning_batch_size)
        routed_evaluation = evaluate_candidates(
            "routed_embedding", routing.reasoning_candidates, ground_truth, args.top_k
        )
        funnel_evaluation = evaluate_funnel(
            embedding_retrieval.diagnostics, routing.diagnostics, batch_plan, routed_evaluation,
            ground_truth, routing.reasoning_candidates,
            cache_pipeline.diagnostics() if cache_pipeline.enabled else None,
        )
        output["benchmark"] = {
            "candidate_evaluations": [
                asdict(evaluate_candidates(
                    "normalized_exact",
                    [item for item in lexical_candidates if item.method == MatchMethod.EXACT_NORMALIZED],
                    ground_truth, args.top_k,
                )),
                asdict(evaluate_candidates("lexical_baseline", lexical_candidates, ground_truth, args.top_k)),
                asdict(evaluate_candidates("embedding", embedding_candidates, ground_truth, args.top_k)),
                asdict(routed_evaluation),
            ],
            "embedding_retrieval_diagnostics": _diagnostics_output(embedding_retrieval.diagnostics, args.pair_evidence),
            "routing_diagnostics": asdict(routing.diagnostics),
            "batch_reasoning_plan": asdict(batch_plan),
            "funnel_evaluation": asdict(funnel_evaluation),
        }
    elif args.match or args.embedding_match:
        if len(profiles) != 2:
            parser.error("matching requires exactly two CSV datasets")
        candidates_for_reasoning: list = []
        if args.embedding_match:
            embedding_retrieval = cache_pipeline.embedding_retrieval(
                profiles[0], profiles[1], model_id=args.embedding_model, top_k=args.top_k,
                minimum_score=args.minimum_score if args.minimum_score is not None else 0.35,
            )
            candidates = embedding_retrieval.candidates
            routing = cache_pipeline.routing(
                candidates, embedding_retrieval.diagnostics,
                minimum_score=args.routing_minimum_score, ambiguity_margin=args.ambiguity_margin,
            )
            candidates_for_reasoning = routing.reasoning_candidates
            batch_plan = batch_reasoning_plan(candidates_for_reasoning, args.reasoning_batch_size)
            output["retrieval_diagnostics"] = _diagnostics_output(embedding_retrieval.diagnostics, args.pair_evidence)
            output["routing_decisions"] = [asdict(item) for item in routing.decisions]
            output["routing_diagnostics"] = asdict(routing.diagnostics)
            output["batch_reasoning_plan"] = asdict(batch_plan)
        else:
            candidates = generate_candidates(
                profiles[0], profiles[1], top_k=args.top_k,
                minimum_score=args.minimum_score if args.minimum_score is not None else 0.6,
            )
            candidates_for_reasoning = candidates
        output["candidates"] = [candidate.__dict__ for candidate in candidates]
        if args.reason:
            source_columns = {column.name: column for column in profiles[0].columns}
            target_columns = {column.name: column for column in profiles[1].columns}
            config = OpenAIConfig.from_environment()
            batch_run = reason_candidate_batches(
                source_columns, target_columns, candidates_for_reasoning, OpenAIReasoningClient(config),
                maximum_batch_size=args.reasoning_batch_size,
                cache=cache_pipeline.file_cache,
                cache_context={"model": config.model, "base_url": config.base_url},
            )
            cache_pipeline.record_batch_reasoning(
                batch_run.diagnostics.cache_hits, batch_run.diagnostics.cache_misses
            )
            reasoning_results = batch_run.results
            output["batch_reasoning_diagnostics"] = asdict(batch_run.diagnostics)
            output["reasoning_results"] = [asdict(result) for result in reasoning_results]
            if args.validate:
                mapping_results = finalize_mappings(source_columns, target_columns, reasoning_results)
                output["mapping_results"] = [asdict(result) for result in mapping_results]
                if args.canonical:
                    output["canonical_schema"] = asdict(build_canonical_schema(mapping_results))
        if args.ground_truth:
            ground_truth = load_ground_truth(args.ground_truth)
            output["evaluation"] = {
                "candidate_recall_at_k": candidate_recall_at_k(candidates, ground_truth)
            }
            if args.evaluate:
                strategy = "embedding" if args.embedding_match else "lexical_baseline"
                output["evaluation"]["candidate_metrics"] = asdict(
                    evaluate_candidates(strategy, candidates, ground_truth, args.top_k)
                )
                if args.embedding_match:
                    output["evaluation"]["routed_candidate_metrics"] = asdict(
                        evaluate_candidates("routed_embedding", candidates_for_reasoning, ground_truth, args.top_k)
                    )
                if args.validate:
                    output["evaluation"]["relationship_metrics"] = asdict(
                        evaluate_relationships(mapping_results, ground_truth)
                    )
    if cache_pipeline.enabled:
        output["cache_diagnostics"] = cache_pipeline.diagnostics()
    serialized = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()

