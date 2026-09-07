# AI Schema & Ontology Mapper

This repository is a local-first schema and ontology mapping project. It profiles
CSV inputs, retrieves lexical or ONNX embedding candidates, and can optionally
obtain structured OpenAI reasoning before deterministic validation. It has no
database or web infrastructure.

## Small-fixture example

Start with `data/examples/basic_customer_client/` to see the same JSON output shape used by the wide fixture on a small 8-by-7 schema comparison. The embedding workflow produces candidates and routing decisions locally; `REASON` means a strong candidate is eligible for optional LLM assessment, not that it is already an accepted mapping.

```bash
uv run python -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --embedding-match
```

The output includes a strong candidate routed for reasoning:

```json
{
  "candidate": {
    "source_column": "annual_income",
    "target_column": "salary",
    "score": 0.884470945654769,
    "method": "EMBEDDING"
  },
  "rank": 1,
  "action": "REASON",
  "reason": "Top candidate meets the configured retrieval score threshold."
}
```

It also keeps lower-ranked candidates visible but avoids unnecessary LLM work when they are clearly weaker:

```json
{
  "candidate": {
    "source_column": "annual_income",
    "target_column": "price_eur",
    "score": 0.7293578515460664,
    "method": "EMBEDDING"
  },
  "rank": 2,
  "top_score_margin": 0.15511309410870266,
  "action": "NO_DECISION",
  "reason": "Runner-up is outside the configured ambiguity margin."
}
```


## Run

```bash
PYTHONPATH=src python3 -m ontology_mapper.cli data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv
```

To save the profiles:

```bash
PYTHONPATH=src python3 -m ontology_mapper.cli data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv --output profiles.json
```

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Synthetic scenarios

- `data/examples/basic_customer_client/` contains the original small 8-by-7 customer/client comparison. Its ground truth is `data/ground_truth/mappings.json`.
- `data/examples/wide_customer_client/` contains two 100-column customer schemas. It has 25 known MATCH pairs, a RELATED pair, a currency CONFLICT, and deliberately type-incompatible distractors for funnel evaluation. Its ground truth is `data/ground_truth/wide_customer_client.json`.

All example files are synthetic. Ground truth is used only for evaluation, not to make profiling or retrieval decisions.


## Baseline matching

Generate deterministic lexical candidates for two datasets:

```bash
PYTHONPATH=src python3 -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --match --ground-truth data/ground_truth/mappings.json
```

Candidate scores are lexical baseline scores, not semantic mapping decisions.

## ONNX embedding matching

Install the PyTorch-free dependencies and run the Hugging Face ONNX model:

```bash
uv sync
uv run python -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --embedding-match --ground-truth data/ground_truth/mappings.json
```

This uses `Xenova/all-MiniLM-L6-v2` from the Hugging Face Hub with ONNX Runtime. It is a candidate-generation step, not a final mapping decision.

PyTorch is intentionally not used because this project only needs local embedding inference, not model training or fine-tuning. Hugging Face tokenizer assets plus ONNX Runtime provide that capability with a smaller dependency footprint, which keeps setup simple for this portfolio project. PyTorch would be a reasonable choice if future work required training, broader model experimentation, or PyTorch-specific tooling.

Embedding retrieval now reports `total_possible_pairs`, `pairs_retained`, `embedding_ranked_pairs`, and `shortlisted_candidates`. Add `--pair-evidence` only when a full retained/excluded pair audit is needed.

For the funnel fixture:

```bash
uv run python -m ontology_mapper.cli \
  data/examples/wide_customer_client/customer_master.csv data/examples/wide_customer_client/customer_platform.csv \
  --embedding-match --top-k 3 --ground-truth data/ground_truth/wide_customer_client.json --evaluate
```

Routing is applied automatically to embedding candidates. By default, it reasons over a top candidate with score at least `0.80`, plus a runner-up when its score is within `0.03`; other candidates are returned as `NO_DECISION`. Override these with `--routing-minimum-score` and `--ambiguity-margin`. `AUTO_ACCEPT` is intentionally not emitted yet, so routing never creates a semantic mapping without the LLM.


## Local cache

Caching is an opt-in local reuse layer. On a first cache-enabled run (a **cold** run), the project profiles the CSVs, performs ONNX embedding retrieval, applies routing, and, when `--reason` is used, calls the LLM for each required batch. It stores the reusable results under `.ontology_mapper_cache/`.

On a later identical cache-enabled run (a **warm** run), the project can reuse:

- Dataset profiles.
- Complete embedding-retrieval and routing artifacts, avoiding ONNX model loading and inference.
- Locally validated batch-reasoning responses, avoiding repeat provider requests for the same batch.

A cache entry is reused only when its relevant evidence and configuration still match. Keys include CSV content, profiling settings, embedding model, retrieval and routing settings, batch prompt/schema version, and—in the reasoning cache—the configured model and base URL. Changing any of those creates a cache miss and recomputes that stage rather than reusing a stale result.

Enable it with `--cache`:

```bash
uv run python -m ontology_mapper.cli source.csv target.csv --embedding-match --cache
```

Without `--cache`, the default behavior is cache-free. `--no-cache` is an explicit bypass option. Cache-enabled JSON output includes `cache_diagnostics` with per-stage hit and miss counts.

The cache is ignored by Git and stores no API key or failed provider response. Deterministic validation, canonical reporting, and evaluation still run fresh from the cached upstream artifacts. It is safe to delete `.ontology_mapper_cache/` when you want a fully cold run.

## OpenAI reasoning

Reasoning is opt-in. The repository-local `.env` file is intentionally blank and ignored by Git. Set real values there when you are ready to make API calls; use `.env.example` as the safe template. Shell environment values, when set, take precedence over `.env`.

After setting real values, run embedding retrieval and structured reasoning:

```bash
uv run python -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --embedding-match --reason
```

The project batches only routed candidates that share a source column. Each request contains that source profile and up to two target candidates by default; change the limit with `--reasoning-batch-size`. A batch response must return exactly one assessment for every requested target, or the command fails rather than silently omitting a candidate. LLM results are provisional and require deterministic validation.

## Deterministic validation and final decisions

Add `--validate` to the opt-in embedding and reasoning workflow to produce final mapping results:

```bash
uv run python -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --embedding-match --reason --validate
```

Validation currently uses narrow, explainable evidence: explicit currency tokens in column names and observed datatype compatibility. Different explicit currencies override a provisional `MATCH` to `CONFLICT`. Different datatypes are warnings rather than automatic conflicts, because identifiers may use different physical representations. Only an unconflicted `MATCH` can be `ACCEPT`; `RELATED` is always `REVIEW`, while `CONFLICT` and `NO_MATCH` are `REJECT`.

## Canonical schema

Add `--canonical` after validation to generate a flat canonical schema and mapping report:

```bash
uv run python -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --embedding-match --reason --validate --canonical
```

Only unambiguous `MATCH` results with `ACCEPT` status become canonical concepts. REVIEW, REJECT, and ambiguous accepted mappings remain in the report but are not asserted as canonical membership.

## Evaluation

Benchmark the three retrieval strategies without an OpenAI request:

```bash
uv run python -m ontology_mapper.cli \
  data/examples/basic_customer_client/customers_a.csv data/examples/basic_customer_client/clients_b.csv \
  --benchmark --ground-truth data/ground_truth/mappings.json
```

For the wide-schema funnel report, run the cache-enabled benchmark twice: the first run records cold misses and the second demonstrates deterministic cache reuse.

```bash
uv run python -m ontology_mapper.cli \
  data/examples/wide_customer_client/customer_master.csv data/examples/wide_customer_client/customer_platform.csv \
  --benchmark --top-k 3 --ground-truth data/ground_truth/wide_customer_client.json --cache
```

The `funnel_evaluation` object summarizes stage reductions, routed MATCH recall, planned provider requests, and cache hit/miss counts. Planned requests are not actual OpenAI calls.

After deliberately configuring OpenAI credentials, run the complete wide-schema evaluation:

```bash
uv run python -m ontology_mapper.cli \
  data/examples/wide_customer_client/customer_master.csv data/examples/wide_customer_client/customer_platform.csv \
  --embedding-match --reason --validate --evaluate --cache \
  --ground-truth data/ground_truth/wide_customer_client.json
```

This opt-in provider-backed command emits final relationship accuracy, MATCH precision/recall/F1, and fixture-limited false-positive rate. It may make billed OpenAI requests.

### Current offline funnel results

| Metric | Result |
| --- | --- |
| Possible source-target pairs | 10,000 |
| Pairs retained after deterministic compatibility filtering | 6,106 (38.94% reduction) |
| Embedding shortlist | 300 candidates (95.09% reduction from retained pairs) |
| Candidates routed for reasoning | 175 (41.67% reduction from shortlist) |
| Planned default provider requests | 100 batches (66.67% reduction from shortlist) |
| Routed expected MATCH coverage | 24 of 25 (recall@3 = 0.96) |
| Missing expected MATCH before reasoning | `customer_id` → `external_customer_id` (string/integer identifier) |

These are offline results from the synthetic wide fixture. Provider-backed final-decision metrics are reported only after running the opt-in command above.
