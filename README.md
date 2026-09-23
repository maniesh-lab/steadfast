# steadfast

A reliability layer for LLM API calls — retries with backoff, per-provider
rate limiting, self-correcting Pydantic-validated structured output, and
correlation-ID structured logging, wrapped around Anthropic, OpenAI, and
Groq's official SDKs behind one provider-agnostic client.

Built as foundational infrastructure, not a standalone demo — reused by every
AI-automation project after it in the same portfolio (a generate→QA→retry
benchmark, an n8n automation layer, an agentic lead-gen pipeline, and a
scheduled ops-bot) instead of each one hand-rolling its own retry, rate-limit,
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
  model's response doesn't validate, the validation error is fed back to the
  model as a correction hint and the call is retried automatically, up to a
  configurable number of attempts
- **Structured JSON logging** (`structlog`) with correlation IDs — every
  retry, rate-limit wait, and validation attempt from one logical call
  shares a single ID
- **20 automated tests**, all running against fake/mocked providers — no
  live API key or network call required to run the suite
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
│   ├── validation.py            # pydantic validate-and-retry loop
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
6. **Validate** (only if a schema was given) — the raw response is checked
   against the schema; on a mismatch, the validation error is folded into
   the next attempt's prompt as a correction hint, and the call is retried
7. **Log** — every step above is logged as structured JSON, tagged with one
   correlation ID per top-level `complete()` call

---

## Install

Published on PyPI as `steadfast-llm` (the plain `steadfast` name was already
taken by an unrelated package — the import name is unaffected):

```bash
pip install steadfast-llm
```

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

## Example Output

```
Plain response: Hello
Validated response: topic='Octopus Intelligence' fact='Some octopus species can solve puzzles and use tools.'
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `httpx` | Async HTTP, shared transport underlying the provider SDKs |
| `tenacity` | Retry policy — exponential backoff with jitter |
| `aiolimiter` | Per-provider async rate limiting |
| `pydantic` | Structured output validation |
| `structlog` | Structured JSON logging with correlation IDs |
| `anthropic` / `openai` / `groq` | Official provider SDKs |
| `pytest` + `pytest-asyncio` | Test suite — 20 tests, all against fakes/mocks |

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

---

## Author

**Manish Pandeya** · [github.com/maniesh-lab](https://github.com/maniesh-lab)