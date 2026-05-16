# AI File Monitoring System

Adaptive, AI-only file monitoring pipeline for futures files. The first phase watches `/data/futures/`, sends incoming CSV rows and the human-readable rule file to an OpenAI-compatible LLM API for validation, optionally loads AI-accepted rows into Oracle SQL, and separates AI-rejected rows with machine-readable reasons.

## Phase 1 workflow

```text
file watcher
  -> agentic validation graph (LangGraph when installed)
  -> rule interpreter agent (LLM)
  -> parallel AI chunk validator agents (LLM)
  -> supervisor reconciliation agent
  -> accepted rows -> optional Oracle SQL/dry-run loader
  -> rejected rows -> rejected CSV + reason JSON + adaptive suggestions
```

The implementation is platform independent and AI-first: it uses Python standard-library polling for file watching, a LangGraph-style workflow for LLM validation, `concurrent.futures` for parallel AI chunk agents, JSON rule files, and an optional Oracle adapter. It does **not** validate rows with local/manual rule logic in the processing path. If no OpenAI-compatible API endpoint/key/model is configured, the processor fails fast instead of silently doing local checks.

## Features

- Continuously watches a directory such as `/data/futures/` for stable CSV files.
- Splits each file into chunks so multiple AI validation agents can check rows in parallel.
- Uses an agentic validation graph: LLM rule interpreter, LLM chunk validators, and supervisor reconciliation.
- Calls qwen, gpt-oss, gemma, or another model through an OpenAI-compatible `/v1/chat/completions` API.
- Sends the JSON rules and row chunks to AI; AI returns row-level accept/reject decisions and reasons.
- Loads only AI-accepted rows when `--load-to-database true` is enabled; one AI-rejected record does not block the rest of the file.
- Writes rejected rows to a temporary/rejected file and writes AI row-level failure reasons to JSON.
- Produces adaptive rule-change suggestions for repeated AI rejection patterns, with human approval required by default.

## Project layout

```text
src/aifilemonitoring/
  agents.py       # shared chunking helpers
  ai_validation.py # LangGraph-style OpenAI-compatible AI validation agents
  adaptive.py     # adaptive pattern detection and rule suggestions
  cli.py          # command-line entrypoint
  llm.py          # OpenAI-compatible chat-completions client
  loaders.py      # dry-run CSV and Oracle loaders
  models.py       # configuration and validation data models
  processor.py    # end-to-end file processing orchestration
  rules.py        # JSON rule loading/config validation helpers
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

The rule loader performs only configuration safety checks (for example, whether a combination expression uses unsupported syntax). Row-level validation is performed by the AI validator agents.

## Run locally on Windows or RHEL Linux

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/RHEL: source .venv/bin/activate
python -m pip install -e .
```

Configure your OpenAI-compatible qwen/gpt-oss/gemma endpoint before processing files, either with environment variables:

```bash
export OPENAI_COMPATIBLE_BASE_URL='https://your-model-gateway.example.com/v1'
export OPENAI_COMPATIBLE_API_KEY='secret'
export OPENAI_COMPATIBLE_MODEL='qwen-or-gpt-oss-or-gemma'
```

Or pass the same details from the terminal:

```bash
ai-file-monitor --config examples/config.local.json \
  --openai-compatible-base-url 'https://your-model-gateway.example.com/v1' \
  --openai-compatible-api-key 'secret' \
  --openai-compatible-model 'qwen-or-gpt-oss-or-gemma' \
  --load-to-database false
```

Process one sample file with AI validation and without database loading:

```bash
mkdir -p data/futures
cp examples/sample_futures.csv data/futures/incoming.csv
ai-file-monitor --config examples/config.local.json --load-to-database false --once data/futures/incoming.csv
```

Process one sample file with AI validation and activate the loader from the terminal. With `dry_run_load_path` configured, this writes AI-accepted rows to a CSV instead of Oracle:

```bash
mkdir -p data/futures
cp examples/sample_futures.csv data/futures/incoming.csv
ai-file-monitor --config examples/config.local.json --load-to-database true --once data/futures/incoming.csv
```

Run continuously with AI validation enabled (AI validation is always required):

```bash
ai-file-monitor --config examples/config.local.json --load-to-database false
```

When `langgraph` is installed, the same AI node sequence is compiled as a `StateGraph`; otherwise the project runs the same AI graph steps locally so development and tests remain lightweight.

## Agentic AI validation behavior

The validation graph runs these agents:

1. **Rule interpreter agent** sends the JSON rule file to the configured LLM and asks it to normalize the rules into validation instructions.
2. **Parallel AI chunk validator agents** send row chunks to the configured OpenAI-compatible qwen/gpt-oss/gemma API and require JSON decisions for every row.
3. **Supervisor reconciliation agent** verifies that every input row has an AI decision and marks rows with missing/failed AI decisions as rejected.

If the API call fails and `ai_fail_closed: true`, affected rows are rejected with an `ai_validation_exception` reason. If API credentials are missing entirely, startup fails fast with a clear configuration error.

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

Database loading is off by default. Turn it on explicitly from the terminal with `--load-to-database true` (or `--load-to-db true`) and turn it off with `--load-to-database false`. For local development, set `dry_run_load_path` to append AI-accepted rows to a CSV instead of Oracle.

## Adaptive AI behavior

The system does not silently weaken controls. It learns from repeated AI failures by generating `*.adaptive_suggestions.json` files beside the reason files. Those suggestions can be reviewed by a human or sent to qwen, gpt-oss, gemma, or another model through the OpenAI-compatible client in `llm.py`.

Recommended operating model:

1. Keep JSON rule definitions as the source of truth for what the AI should validate.
2. Let AI validation detect bad data and explain row-level failures.
3. Let adaptive analysis detect repeated rejection patterns.
4. Ask the OpenAI-compatible model gateway to explain whether a business rule may have changed.
5. Require data-owner approval before updating the JSON rule file.
6. Version-control each rule-file change.

## Development checks

```bash
python -m pytest
```
