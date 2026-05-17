# AI File Monitoring System

Adaptive, AI-only file monitoring pipeline for futures files. The first phase watches `/data/futures/`, sends incoming CSV rows and the human-readable rule file to an OpenAI-compatible LLM API for validation, optionally loads AI-accepted rows into Oracle SQL, and separates AI-rejected rows with machine-readable reasons.

## Phase 1 workflow

```text
file watcher
  -> native LangGraph StateGraph
  -> rule interpreter agent (LLM)
  -> chunk planner agent
  -> Send(chunk_1) -> validator agent (LLM)
  -> Send(chunk_2) -> validator agent (LLM)
  -> Send(chunk_N) -> validator agent (LLM)
  -> supervisor agent verifies all chunk results
  -> accepted rows -> optional Oracle SQL/dry-run loader
  -> rejected rows -> rejected CSV + reason JSON
  -> adaptive rule-change agent tracks repeated failures
  -> approval notification email/outbox
  -> approved proposal can update rules and optional DB schema
```

The implementation is platform independent and AI-first. It uses native LangGraph `StateGraph` + `Send` map/reduce fan-out for chunk validation, JSON rule files, an OpenAI-compatible LLM API, and an optional Oracle adapter. It does **not** validate rows with local/manual rule logic in the processing path. If no OpenAI-compatible API endpoint/key/model is configured, the processor fails fast instead of silently doing local checks.

## Features

- Continuously watches a directory such as `/data/futures/` for stable CSV files.
- Splits each file into chunks and dispatches each chunk to a LangGraph validator agent through `Send`.
- Uses native LangGraph agents: LLM rule interpreter, chunk planner, parallel LLM chunk validators, and supervisor.
- Calls qwen, gpt-oss, gemma, or another model through an OpenAI-compatible `/v1/chat/completions` API.
- Sends the JSON rules and row chunks to AI; AI returns row-level accept/reject decisions and reasons.
- Loads only AI-accepted rows when `--load-to-database true` is enabled.
- Tracks repeated validation failures over a configurable number of consecutive days.
- Uses an adaptive AI rule-change agent to propose updated rule JSON and optional DB schema SQL.
- Sends/writes approval notifications for rule and schema changes.
- Applies approved rule changes only when the user passes `--approve-rule-change <proposal_id>`.
- Applies approved DB schema changes only when the user also passes `--apply-db-schema-change true`.

## Project layout

```text
src/aifilemonitoring/
  agents.py         # shared chunking helpers
  ai_validation.py  # native LangGraph AI validation graph
  adaptive_agent.py # adaptive failure tracking, proposal, notification, approval
  adaptive.py       # legacy suggestion utilities
  cli.py            # command-line entrypoint
  llm.py            # OpenAI-compatible chat-completions client
  loaders.py        # dry-run CSV and Oracle loaders
  models.py         # configuration and validation data models
  processor.py      # end-to-end file processing orchestration
  rules.py          # JSON rule loading/config validation helpers
  watcher.py        # cross-platform continuous polling watcher
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

The rule loader performs only configuration safety checks. Row-level validation is performed by the AI validator agents.

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

Run continuously with AI validation enabled:

```bash
ai-file-monitor --config examples/config.local.json --load-to-database false
```

## Adaptive rule-change workflow

Enable adaptive rule-change proposals when you want the system to learn from repeated failures:

```bash
ai-file-monitor --config examples/config.local.json \
  --adaptive-rule-change true \
  --adaptive-day-threshold 10 \
  --notification-email-to data-owner@example.com
```

What happens:

1. AI validation rejects rows with structured reasons.
2. `AdaptiveRuleAgent` records failure patterns by day in `adaptive_history_file`.
3. If the same failure pattern occurs for the configured consecutive-day threshold, the adaptive AI agent asks the LLM to propose updated rules.
4. The proposal is written to `rule_change_dir/<proposal_id>.json`.
5. A notification is sent by SMTP when configured; otherwise an `.eml` file is written to `notification_outbox_dir`.
6. Nothing is changed automatically until a user approves the proposal.

Approve and apply a rule proposal:

```bash
ai-file-monitor --config examples/config.local.json --approve-rule-change <proposal_id>
```

If the AI proposal includes DB schema changes, the system writes `<proposal_id>.schema.sql` and sends/writes a schema review notification. To apply those schema changes to Oracle as well:

```bash
ai-file-monitor --config examples/config.local.json \
  --approve-rule-change <proposal_id> \
  --apply-db-schema-change true
```

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

## Development checks

```bash
python -m pytest
```
