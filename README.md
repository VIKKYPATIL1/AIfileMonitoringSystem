# AI File Monitoring System

Adaptive, AI-assisted file monitoring pipeline for futures files. The first phase watches `/data/futures/`, validates each incoming CSV against a human-readable rule file, optionally loads valid rows into Oracle SQL, and separates failed rows with machine-readable reasons.

## Phase 1 workflow

```text
file watcher
  -> native LangGraph agentic validation graph
  -> rule interpreter agent
  -> parallel AI chunk validator agents
  -> deterministic guardrail agent
  -> supervisor reconciliation agent
  -> accepted rows -> optional Oracle SQL/dry-run loader
  -> rejected rows -> rejected CSV + reason JSON
  -> cumulative adaptive history + AI rule-change review
  -> AI-planned matplotlib analytics
```

The implementation is intentionally platform independent: it uses Python standard-library polling for file watching, a LangGraph-style agent workflow for AI validation, `concurrent.futures` for parallel chunk agents, JSON rule files, and an optional Oracle adapter.

## Features

- Continuously watches a directory such as `/data/futures/` for stable CSV files.
- Splits each file into chunks so multiple AI validation agents can check rows in parallel.
- Uses an agentic validation graph: rule interpreter, AI chunk validators, deterministic guardrail, and supervisor reconciliation.
- Calls qwen, gpt-oss, gemma, or another model through an OpenAI-compatible `/v1/chat/completions` API.
- Applies deterministic column guardrails such as required, type, min/max, allowed values, regex, and length.
- Applies deterministic cross-column guardrails through a safe expression evaluator instead of unsafe `eval`.
- Loads only valid rows when `--load-to-database true` is enabled; one bad record does not block the rest of the file.
- Writes rejected rows to a temporary/rejected file and writes row-level failure reasons to JSON.
- Produces adaptive rule-change suggestions for repeated patterns across files, with human approval required by default.
- Maintains cumulative adaptive history so one large bad file does not look like many days of repeated failures.
- Uses the configured OpenAI-compatible model to review adaptive rule-change candidates and choose useful rejection analytics charts.
- Writes analytics JSON and matplotlib PNG charts for accepted/rejected counts, failed rules, failed columns, and rule/column concentration.
- Supports qwen, gpt-oss, gemma, or other OpenAI-compatible model gateways for validation and review assistance.

## Project layout

```text
src/aifilemonitoring/
  agents.py       # deterministic parallel validation worker pool
  ai_validation.py # LangGraph-style OpenAI-compatible AI validation agents
  adaptive.py     # cumulative adaptive pattern detection and AI rule suggestions
  analytics.py    # AI-planned matplotlib rejection analytics
  cli.py          # command-line entrypoint
  llm.py          # OpenAI-compatible chat-completions client
  loaders.py      # dry-run CSV and Oracle loaders
  models.py       # configuration and validation data models
  processor.py    # end-to-end file processing orchestration
  rules.py        # JSON rule loading and rule engine
  watcher.py      # cross-platform continuous polling watcher
examples/
  config.local.json
  rules.futures.json
  sample_futures.csv
tests/
```

## Rule file format

Rules are JSON so humans and LLMs can read and edit them easily. See [`examples/rules.futures.json`](examples/rules.futures.json).

```json
{
  "columns": {
    "quantity": {"type": "integer", "required": true, "min": 1, "max": 100000},
    "symbol": {"type": "string", "required": true, "allowed": ["ES", "NQ", "CL", "GC"]}
  },
  "combinations": [
    {
      "name": "energy_contract_price_band",
      "expression": "symbol != 'CL' or price <= 500",
      "reason": "Crude oil futures price is outside approved operational band"
    }
  ],
  "adaptive": {"suggestion_threshold": 10, "auto_apply": false, "review_required": true}
}
```

Supported column rule keys:

| Key | Purpose |
| --- | --- |
| `required` | Rejects blank values when `true`. |
| `nullable` | Allows blanks when `true`. Defaults to the opposite of `required`. |
| `type` | `string`, `integer`, `decimal`, `date`, or `datetime`. |
| `format` | Date/datetime parsing format, for example `%Y-%m-%d`. |
| `min` / `max` | Numeric or date boundaries. |
| `allowed` | List of approved values. |
| `regex` | Full-match regular expression. |
| `min_length` / `max_length` | String length limits. |

Combination expressions may use column names, constants, boolean operators, comparisons, and list/tuple membership. Function calls, imports, attributes, and arithmetic are blocked.

## Run locally on Windows or RHEL Linux

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/RHEL: source .venv/bin/activate
python -m pip install -e .
```

Process one sample file without database loading:

```bash
mkdir -p data/futures
cp examples/sample_futures.csv data/futures/incoming.csv
ai-file-monitor --config examples/config.local.json --load-to-database false --once data/futures/incoming.csv
```

Process one sample file and activate the loader from the terminal. With `dry_run_load_path` configured, this writes accepted rows to a CSV instead of Oracle:

```bash
mkdir -p data/futures
cp examples/sample_futures.csv data/futures/incoming.csv
ai-file-monitor --config examples/config.local.json --load-to-database true --once data/futures/incoming.csv
```

Run continuously with deterministic guardrails only:

```bash
ai-file-monitor --config examples/config.local.json --load-to-database false
```

Run continuously with the agentic AI validation graph enabled:

```bash
export OPENAI_COMPATIBLE_BASE_URL='https://your-model-gateway.example.com/v1'
export OPENAI_COMPATIBLE_API_KEY='secret'
export OPENAI_COMPATIBLE_MODEL='qwen-or-gpt-oss-or-gemma'
ai-file-monitor --config examples/config.local.json --ai-validation --load-to-database false
```

`langgraph` is a runtime dependency. The validator compiles the validation node sequence as a `StateGraph`; the lightweight local graph path remains only as a defensive fallback for constrained test environments.


## Agentic AI validation modes

Set `ai_validation_enabled` or pass `--ai-validation` to use AI in the validation path instead of only deterministic checks. The graph runs these agents:

1. **Rule interpreter agent** converts the JSON rule file into explicit validation instructions for the model.
2. **Deterministic guardrail agent** evaluates the same rows with the local rule engine so required controls are never silently skipped.
3. **Parallel AI chunk validator agents** send row chunks to the configured OpenAI-compatible qwen/gpt-oss/gemma API.
4. **Supervisor reconciliation agent** merges results. In the default `assistive` mode, deterministic failures are always preserved and AI may add additional data-quality failures. In `authoritative` mode, the AI decision is used directly.

If AI validation is enabled but no OpenAI-compatible API is configured, `ai_fail_closed: true` rejects rows with an `ai_client_unavailable` reason instead of silently loading unverified data.

## Oracle loading

Install the optional Oracle dependency and configure connection settings with JSON or environment variables:

```bash
python -m pip install -e '.[oracle]'
export ORACLE_DSN='host:1521/service'
export ORACLE_USER='app_user'
export ORACLE_PASSWORD='secret'
export ORACLE_TABLE='FUTURES_TRADES'
ai-file-monitor --config examples/config.local.json --load-to-database true
```

Database loading is off by default. Turn it on explicitly from the terminal with `--load-to-database true` (or `--load-to-db true`) and turn it off with `--load-to-database false`. For local development, set `dry_run_load_path` to append accepted rows to a CSV instead of Oracle.

## Adaptive AI behavior

The system does not silently weaken controls. It learns from repeated failures by maintaining a cumulative `adaptive_history.json` file and generating `*.adaptive_suggestions.json` only when the same failure pattern appears in enough distinct files. `suggestion_threshold` means **files with the same failure pattern**, not total rejected rows.

When an OpenAI-compatible client is configured, the adaptive advisor asks the model to classify each recurring pattern as likely bad source data, changed business rule, or possible database schema change. The AI response is written into the adaptive suggestion JSON with proposed rule changes, proposed schema changes, and approval questions.

Each adaptive suggestion also creates reviewer-friendly table artifacts beside the JSON:

- `*.adaptive_suggestions_table.csv`
- `*.adaptive_suggestions_table.png`

The table columns are `Column name`, `Accepted format`, `Received value from file`, `Description`, and `New change needed if accepted`, so reviewers can inspect the business decision without reading the full JSON payload.

Recommended operating model:

1. Keep deterministic rules as the source of truth.
2. Let adaptive analysis detect repeated rejection patterns across files.
3. Ask the OpenAI-compatible model gateway to explain whether a business rule or schema may have changed.
4. Require data-owner approval before updating the JSON rule file.
5. Version-control each rule-file change.

## Visual analytics

For files with rejected records, the processor writes an analytics JSON report and matplotlib charts under `analytics_dir`. The AI analytics agent receives accepted/rejected counts, failed-rule counts, failed-column counts, and sample errors, then chooses chart types from a safe renderer allow-list:

- `status_pie`
- `failed_rules_bar`
- `failed_columns_bar`
- `rule_by_column_heatmap`

If no OpenAI-compatible API is configured, the report records `llm_status: not_configured` and renders a conservative default chart set.

## Development checks

```bash
python -m pytest
```
