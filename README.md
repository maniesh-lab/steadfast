# steadfast

[![CI](https://github.com/maniesh-lab/steadfast/actions/workflows/ci.yml/badge.svg)](https://github.com/maniesh-lab/steadfast/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/steadfast-llm)](https://pypi.org/project/steadfast-llm/)

A reliability layer for LLM API calls — retries with backoff, per-provider
rate limiting, self-correcting Pydantic-validated structured output, and
correlation-ID structured logging, wrapped around Anthropic, OpenAI, and
Groq's official SDKs behind one provider-agnostic client.

Built as foundational infrastructure, not a standalone demo — reused by every
AI-automation project after it in the same portfolio (an n8n automation
layer, an agentic lead-gen pipeline, and a scheduled ops-bot) instead of each one hand-rolling its own retry, rate-limit,
and validation logic.

---

## What it does

Wraps calls to LLM providers with the operational concerns a production AI
backend needs and otherwise has to hand-roll: retrying transient failures
without wasting retries on permanent ones, respecting per-provider rate
limits, validating that structured output actually matches the shape it's
supposed to (and asking the model to fix its own mistake if it doesn't), and
tagging every log line from one logical call with a shared correlation ID so
the full story of a single request can be traced after the fact.

---

## Features

- **Provider-agnostic client** — one `complete()` call works identically
  across Anthropic, OpenAI, and Groq
- **Retry policy** (via `tenacity`) that only retries transient failures
  (rate limits, 5xx errors, connection drops) — permanent failures (bad API
  key, malformed request) raise immediately instead of burning retry
  attempts on something that will never succeed
- Each provider's own built-in retry is disabled (`max_retries=0`) so retry
  behavior is decided in exactly one place, consistently, across all three
  providers
- **Per-provider rate limiting** (`aiolimiter`) — Anthropic, OpenAI, and Groq
  each get independent limits, configured separately
- **Self-correcting structured output** — pass a Pydantic schema, and if the
  model's response doesn't validate, the problems are fed back to the model
  (along with its own previous reply) and the call is retried automatically,
  up to `max_attempts` total tries (default 3, settable per call)
- **Tolerant JSON extraction** — replies wrapped in markdown code fences or
  surrounded by chatter are cleaned up before validation, so they don't waste
  a retry
- **Custom validators** — pass `validators=[fn, ...]` to enforce rules a
  schema can't express (a quote must appear verbatim in the source, deadlines
  must be working days, totals must add up). Each function takes the parsed
  object and returns `None` if fine, or a string describing the problem; that
  string goes back to the model on retry
- **Benchmarked** — a reproducible no-retry vs retry comparison ships in
  `benchmark/` (results below)
- **Structured JSON logging** (`structlog`) with correlation IDs — every
  retry, rate-limit wait, and validation attempt from one logical call
  shares a single ID. Pass your own with `correlation_id=...`, or call it
  inside code that already bound one (like a web request) and that ID is
  reused and left in place afterwards
- **42 automated tests**, all running against fake/mocked providers — no
  live API key or network call required to run the suite
- **CI on every push** — lint, tests, and a package build on Python 3.10–3.13
- **Zero-cost to try** — the test suite and the example script both work
  entirely on free-tier access (Groq)

---

## Project Structure

```
steadfast/
│
├── steadfast/
│   ├── __init__.py
│   ├── client.py                # SteadfastClient — wires everything together
│   ├── exceptions.py            # custom exception hierarchy
│   ├── retry.py                 # tenacity retry policy
│   ├── rate_limit.py            # per-provider rate limiting
│   ├── validation.py            # parse → schema → custom validators → feedback loop
│   ├── logging_utils.py         # structlog + correlation ID context
│   └── providers/
│       ├── base.py              # abstract provider interface
│       ├── anthropic_provider.py
│       ├── openai_provider.py
│       └── groq_provider.py
│
├── tests/
│   ├── conftest.py              # shared fact_schema fixture
│   ├── test_retry.py
│   ├── test_rate_limit.py
│   ├── test_validation.py
│   ├── test_logging.py
│   └── test_client.py
│
├── examples/
│   └── basic_usage.py           # plain call + schema-validated call, live against Groq
│
├── benchmark/
│   ├── tasks.py                 # 6 extraction tasks + rule-based validators
│   ├── run_benchmark.py         # no-retry vs retry comparison, live against Groq
│   └── results.md               # latest results
│
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Register** — a provider (Anthropic/OpenAI/Groq) is created and
   registered with the client, along with its rate limit
2. **Call** — `client.complete(provider_name, prompt, schema=...)` is called
   with a prompt and, optionally, a Pydantic schema describing the expected
   response shape
3. **Rate limit** — the call waits, if needed, for a slot to open under that
   provider's configured rate
4. **Request** — the prompt is sent through the provider's real SDK;
   SDK-level retries are disabled so nothing retries invisibly underneath
   Steadfast
5. **Retry** — if the call fails with a transient error, it's retried
   automatically with exponential backoff and jitter, up to a configurable
   number of attempts; permanent errors raise immediately instead
6. **Validate** (only if a schema was given) — the JSON is extracted from the
   raw response, checked against the schema, then run through any custom
   validators; on any failure, the model's previous reply and the list of
   problems are folded into the next attempt's prompt, and the call is
   retried (up to `max_attempts` total tries)
7. **Log** — every step above is logged as structured JSON, tagged with one
   correlation ID per top-level `complete()` call

---

## Install

Published on PyPI as `steadfast-llm` (the plain `steadfast` name was already
taken by an unrelated package — the import name is unaffected):

```bash
pip install steadfast-llm
```

Requires Python 3.10 or newer.

```python
from steadfast.client import SteadfastClient
```

## How to Run (from source)

**1. Clone the repo**
```bash
git clone https://github.com/maniesh-lab/steadfast
cd steadfast
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -e ".[providers,dev]"
```

**4. Set up environment variables**
```bash
cp .env.example .env
```
Add a free Groq API key from [console.groq.com](https://console.groq.com) to
`.env` (Anthropic/OpenAI keys are optional — only needed if you use those
providers instead).

**5. Run the example**
```bash
python examples/basic_usage.py
```

**6. Run the test suite**
```bash
pytest tests/ -v
```

---

## Example Usage

```python
import asyncio
from pydantic import BaseModel
from steadfast.client import SteadfastClient
from steadfast.providers.groq_provider import GroqProvider


class Fact(BaseModel):
    topic: str
    fact: str


async def main():
    client = SteadfastClient()
    client.register(GroqProvider(), rate=30, period=60)

    # Plain text — no schema
    answer = await client.complete("groq", "Say hello in one word.")

    # Structured, self-correcting output
    fact = await client.complete(
        "groq",
        'Give me a fact about octopuses as JSON: {"topic": "...", "fact": "..."}',
        schema=Fact,
    )

asyncio.run(main())
```

### Custom validators and attempts

```python
fact = await client.complete(
    "groq",
    prompt,
    schema=Fact,
    max_attempts=3,  # total tries, including the first
    validators=[
        lambda f: None if f.fact in source_text
        else "fact must be copied verbatim from the source text",
    ],
)
```

If every attempt fails, `ValidationRetryExceeded` is raised, carrying
`attempts`, `last_response` (the model's last reply), and `errors` (the
problems from the last attempt).

### Correlation IDs

```python
# Tag this call's log lines with your own ID (e.g. your web request's ID)
answer = await client.complete("groq", prompt, correlation_id=request_id)
```

If you leave it out, an ID already bound with `structlog.contextvars` is
reused, otherwise a new one is generated.

## Example Output

```
Plain response: Hello
Validated response: topic='Octopus Intelligence' fact='Some octopus species can solve puzzles and use tools.'
```

---

## Benchmark

Does validate-and-retry actually help? `benchmark/run_benchmark.py` runs six
extraction tasks (invoice, verbatim quotes, review score, action items, order
total, contact) three times each, with and without retry, using identical
schemas and validators. Validators state the rule that was broken, never the
correct answer.

Results with `openai/gpt-oss-20b` on Groq (2026-10-10):

| Mode | Passed | Pass rate | Avg API calls |
|---|---|---|---|
| No retry (`max_attempts=1`) | 15 / 18 | 83% | 1.11 |
| With retry (`max_attempts=3`) | 18 / 18 | 100% | 1.56 |

How to read this:

- **The gain came from one task.** "Action items" passed 0/3 without retry and
  3/3 with it. The other five tasks passed first try every time, so retry had
  nothing to fix there.
- **Retry cost about 40% more API calls** on average (1.56 vs 1.11).
- **Latency is not reported as a clean cost.** The benchmark ran under a
  12-calls-per-minute rate limit, and time spent waiting on the limiter counts
  toward each request's duration, so the raw timings overstate the cost of
  retrying.
- **It is a small sample:** 18 requests per mode, one model, one run. It is a
  signal, not proof. A weaker model would likely show a larger gap.

Reproduce it (needs a Groq key in `.env`):

```bash
python benchmark/run_benchmark.py            # optional: --trials 5 --model <name>
```

Full output is in [`benchmark/results.md`](benchmark/results.md).

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `tenacity` | Retry policy — exponential backoff with jitter |
| `aiolimiter` | Per-provider async rate limiting |
| `pydantic` | Structured output validation |
| `structlog` | Structured JSON logging with correlation IDs |
| `anthropic` / `openai` / `groq` | Official provider SDKs |
| `pytest` + `pytest-asyncio` | Test suite — 42 tests, all against fakes/mocks |
| `ruff` + GitHub Actions | Lint and CI |

---

## Use Case

Built as the reliability layer every other AI-automation project in this
portfolio sits on top of, rather than each one hand-rolling its own retry,
rate-limit, and validation logic. Useful anywhere an app calls an LLM API and
needs it to fail gracefully, respect rate limits, and produce output that
actually matches an expected shape — instead of trusting the model to get it
right on the first try, every time.

---

## Known Limitations

Listed deliberately, not hidden:

- **Only `max_tokens` is forwarded to the underlying SDK call.**
  `complete()` accepts arbitrary `**kwargs` (e.g. `temperature`), but the
  current provider adapters only read `max_tokens` out of that dictionary —
  other settings are silently dropped rather than passed through. Easy to
  extend, just not wired up yet for anything beyond `max_tokens`.
- **No streaming support.** `complete()` returns a full response, not a
  token stream — not currently needed by anything Steadfast is used for, but
  a real constraint if a future use case wants incremental output.
- **`@with_retry()` is applied fresh on every `complete()` call**, rather
  than being built once and reused. Functionally correct, a minor
  inefficiency a very close review would flag.
- **Retry/backoff timing in `client.py` isn't configurable per-call** — it
  currently uses `retry.py`'s defaults (5 attempts, 1–30s backoff) rather
  than exposing those settings through `complete()` itself.
- **Only three providers implemented** (Anthropic, OpenAI, Groq) — adding
  another means writing one more adapter file following the same
  `BaseProvider` contract, not a structural change.
- **No persistent/file logging configured out of the box** —
  `configure_logging()` prints structured JSON to stdout; wiring that into a
  file or log aggregator is left to whoever integrates Steadfast into a
  larger system.
- **Feedback goes into a single prompt, not multi-turn messages.** The model's
  previous reply (truncated to 2,000 characters) is pasted into the next
  prompt rather than sent as a separate assistant message, because every
  provider adapter takes one prompt string.
- **Custom validators are synchronous.** A validator must return `None` or a
  string. One that raises is treated as a bug in the validator and propagates
  rather than being retried.
- **Truncated replies get no special handling.** A reply cut off by
  `max_tokens` fails as invalid JSON and uses up an attempt, because the
  provider adapters don't expose `finish_reason` yet.
- **The benchmark is small** (see above) and covers one model.

---

## Author

**Manish Pandeya** · [github.com/maniesh-lab](https://github.com/maniesh-lab)