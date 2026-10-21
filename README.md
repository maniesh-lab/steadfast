# steadfast

[![CI](https://github.com/maniesh-lab/steadfast/actions/workflows/ci.yml/badge.svg)](https://github.com/maniesh-lab/steadfast/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/steadfast-llm)](https://pypi.org/project/steadfast-llm/)
[![Python](https://img.shields.io/pypi/pyversions/steadfast-llm)](https://pypi.org/project/steadfast-llm/)

A reliability layer for LLM API calls. Wrap your OpenAI, Anthropic, or Groq
calls in Steadfast and they retry when the API is busy, stay under rate
limits, check that the model's output has the right shape (and ask it to fix
it when it doesn't), fall back to another provider if one goes down, and log
every step with one ID per request.

It also handles conversations and tool calling in one format for all three
providers, so it can sit underneath an agent.

---

## The problem

When an app talks to an LLM API, the same things go wrong again and again:

| What goes wrong | What Steadfast does |
|---|---|
| The API is busy (429), down (5xx), or the connection drops | Waits and tries again, longer each time. Permanent errors (bad key, bad request) fail right away instead of wasting retries |
| You send requests faster than your plan allows | Makes calls wait in line, with a separate limit per provider |
| The model returns broken JSON, the wrong fields, or a broken tool call | Tells the model exactly what was wrong and asks again |
| One provider keeps failing | Moves on to the next provider you listed |
| Something broke and you can't tell what happened | Logs every call, retry, and wait as JSON, all tagged with one correlation ID |
| You don't know what it's costing you | Counts input and output tokens per provider |

---

## Install

```bash
pip install "steadfast-llm[groq]"        # or [openai], [anthropic], or [providers] for all three
```

Requires Python 3.10+. The import name is `steadfast` (the plain `steadfast`
name on PyPI belongs to an unrelated package).

---

## Quick start

```python
import asyncio
from steadfast import SteadfastClient
from steadfast.providers.groq_provider import GroqProvider


async def main():
    client = SteadfastClient()
    client.register(GroqProvider(), rate=30, period=60)  # 30 calls per minute

    answer = await client.complete("groq", "Say hello in one word.")
    print(answer)

asyncio.run(main())
```

API keys come from the usual environment variables (`GROQ_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or `GroqProvider(api_key=...)`. Pick a
model with `GroqProvider(model="...")`.

---

## Structured output

Pass a Pydantic model and you get a validated object back. If the reply isn't
valid JSON, has the wrong fields, or fails one of your checks, Steadfast sends
the model its own reply plus the list of problems and asks again, up to
`max_attempts` total tries.

```python
from pydantic import BaseModel

class Fact(BaseModel):
    topic: str
    fact: str

fact = await client.complete(
    "groq",
    'Give me a fact about octopuses as JSON: {"topic": "...", "fact": "..."}',
    schema=Fact,
    system="Answer only in JSON.",
    max_attempts=3,
    validators=[
        # Rules a schema can't express. Return None if fine, or a string saying
        # what's wrong. That string goes back to the model. Can also be async.
        lambda f: None if len(f.fact) < 200 else "fact must be under 200 characters",
    ],
)
```

If every attempt fails you get `ValidationRetryExceeded`, with `attempts`,
`errors` (the last problems), and `last_response` (the model's last reply).
JSON wrapped in code fences or surrounded by chatter is cleaned up first, so
it doesn't waste a try. A reply cut off by `max_tokens` gets a "be shorter"
hint.

---

## Conversations and tool calling

`chat()` takes a list of messages and returns an `LLMResponse`, the same shape
for every provider:

```python
response = await client.chat("groq", [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi!"},
])
response.text           # the reply
response.tool_calls     # tools the model wants to run ([] if none)
response.finish_reason  # "stop", "length", or "tool_calls"
response.usage          # Usage(input_tokens=..., output_tokens=...)
response.provider       # which provider answered
```

Messages and tools use OpenAI's format everywhere. For Anthropic, Steadfast
translates them (system prompt, tool calls, and tool results) both ways.

A tool is a function the model can ask you to run. Describe it, pass it in,
run it when asked, and send the result back:

```python
from pydantic import BaseModel
from steadfast import function_tool, tool_result

class OrderArgs(BaseModel):
    order_id: str

tools = [function_tool("get_order_status", "Look up an order by ID.", OrderArgs)]
messages = [{"role": "user", "content": "Where is order A100?"}]

for _ in range(5):  # always cap the steps
    response = await client.chat("groq", messages, tools=tools)
    if not response.tool_calls:
        print(response.text)
        break
    messages.append(response.to_message())          # keep the model's request
    for call in response.tool_calls:
        args = OrderArgs(**call.arguments)           # already parsed from JSON
        messages.append(tool_result(call, look_up(args.order_id)))
```

If the model sends tool-call arguments that aren't valid JSON, Steadfast asks
it to try again, so the `arguments` you get are always a parsed dict. The full
runnable version is in [`examples/tool_calling.py`](examples/tool_calling.py).
`tool_choice` (`"auto"`, `"required"`, `"none"`, or a specific tool) is
supported too.

---

## Fallbacks

```python
client.register(GroqProvider(), rate=30, period=60)
client.register(OpenAIProvider(model="gpt-4o-mini"), rate=60, period=60)

answer = await client.complete("groq", prompt, fallbacks=["openai"])
```

Want two models from the same provider? Register them under different names:
`client.register(GroqProvider(model="..."), name="groq-small")`.

Each provider gets its full set of retries first. If every provider fails you
get `AllProvidersFailed`, whose `errors` lists what went wrong with each one.

---

## Retries and rate limits

```python
client = SteadfastClient(
    max_retries=5,   # tries per provider for transient errors, first try included
    min_wait=1.0,    # backoff starts here (seconds) ...
    max_wait=30.0,   # ... and never goes above this
)
client.register(GroqProvider(timeout=60), rate=30, period=60)
```

What counts as transient: rate limits (429), timeouts (408 and request
timeouts), conflicts (409), server errors (5xx, including Anthropic's 529
"overloaded"), dropped connections, and Groq's `tool_use_failed`. Everything
else raises `NonRetryableProviderError` immediately. When a provider sends a
`Retry-After` header, Steadfast waits at least that long (up to 60 seconds).

The SDKs' own hidden retries are switched off (`max_retries=0`), so all retry
decisions happen in one place and every retry shows up in the logs.

Any other settings (`temperature`, `max_tokens`, `stop`, `top_p`, ...) are
passed straight through to the provider.

---

## Token usage

```python
client.usage["groq"]                  # Usage(input_tokens=..., output_tokens=...)
client.total_usage.total_tokens       # across all providers
```

Counted for every call, including retries caused by bad output, since those
cost tokens too.

---

## Logging and correlation IDs

```python
from steadfast import configure_logging
configure_logging()  # JSON logs to stdout
```

Every log line from one call shares a correlation ID: the call itself, each
retry, rate-limit waits, validation failures, fallbacks, and a final line with
latency, tokens, and finish reason. Pass your own with
`correlation_id=request_id`, or call Steadfast inside code that already bound
one with `structlog.contextvars` (like a web request) and that ID is reused
and left in place afterwards.

---

## Errors you can get

| Exception | When |
|---|---|
| `RetryableProviderError` | Transient error that kept happening after every retry (has `retry_after` if the provider sent one) |
| `NonRetryableProviderError` | Permanent error: bad key, bad request, unknown model |
| `AllProvidersFailed` | You passed `fallbacks` and every provider failed (`errors` has each one) |
| `ValidationRetryExceeded` | Output still invalid after `max_attempts` tries |

All of them inherit from `SteadfastError`, and the provider errors from
`ProviderError` (with `provider` and `original_exception`).

---

## How one call works

From the outside in:

1. **Correlation ID**: one ID is bound for every log line of this call
2. **Validation loop**: bad output goes back to the model with the problems listed, up to `max_attempts`
3. **Fallback chain**: if a provider keeps failing, try the next one
4. **Transport retry**: transient errors wait (backoff plus jitter, or `Retry-After`) and try again
5. **Rate limiter**: wait for a free slot under that provider's limit
6. **Provider adapter**: the real SDK call, with the answer translated into an `LLMResponse`

---

## Benchmark

Does validate-and-retry actually help? `benchmark/run_benchmark.py` runs six
extraction tasks (invoice, verbatim quotes, review score, action items, order
total, contact) three times each, with and without retry, using the same
schemas and validators. Validators state the rule that was broken, never the
correct answer.

Results with `openai/gpt-oss-20b` on Groq (2026-10-10, measured with v0.2):

| Mode | Passed | Pass rate | Avg API calls |
|---|---|---|---|
| No retry (`max_attempts=1`) | 15 / 18 | 83% | 1.11 |
| With retry (`max_attempts=3`) | 18 / 18 | 100% | 1.56 |

How to read this:

- **The gain came from one task.** "Action items" passed 0/3 without retry and
  3/3 with it. The other five passed first try every time.
- **Retry cost about 40% more API calls** on average (1.56 vs 1.11).
- **Latency isn't reported as a clean cost**: the run used a
  12-calls-per-minute limit, and time spent waiting on the limiter inflates
  the timings.
- **It's a small sample**: 18 requests per mode, one model, one run. A signal,
  not proof.
- **Measured before v0.3**, when feedback was pasted into one prompt instead
  of sent as conversation turns. Rerun it to get v0.3 numbers.

```bash
python benchmark/run_benchmark.py            # optional: --trials 5 --model <name>
```

Full output is in [`benchmark/results.md`](benchmark/results.md).

---

## Project structure

```
steadfast/
├── steadfast/
│   ├── __init__.py              # public API: SteadfastClient, errors, helpers
│   ├── client.py                # SteadfastClient: chat(), complete(), fallbacks, usage
│   ├── types.py                 # LLMResponse, ToolCall, Usage, tool_result()
│   ├── tools.py                 # function_tool() for describing tools
│   ├── exceptions.py            # error types
│   ├── retry.py                 # backoff + Retry-After policy (tenacity)
│   ├── rate_limit.py            # per-provider rate limiting (aiolimiter)
│   ├── validation.py            # parse -> schema -> validators -> feedback
│   ├── logging_utils.py         # structlog setup + correlation IDs
│   ├── py.typed                 # marks the package as typed
│   └── providers/
│       ├── __init__.py
│       ├── base.py              # adapter contract + shared error rules
│       ├── _openai_compatible.py  # shared OpenAI-format code (OpenAI, Groq)
│       ├── openai_provider.py
│       ├── groq_provider.py
│       └── anthropic_provider.py  # translates to and from Anthropic's format
├── tests/                       # 97 tests, all against fakes (no API key needed)
├── examples/
│   ├── basic_usage.py           # plain + validated call
│   └── tool_calling.py          # a small agent loop
├── benchmark/                   # no-retry vs retry comparison
├── .github/workflows/ci.yml     # lint, tests, build, oldest-versions check
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Development

```bash
git clone https://github.com/maniesh-lab/steadfast
cd steadfast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[providers,dev]"

pytest                    # 97 tests, no API key needed
ruff check .              # lint
ruff format .             # format the code (CI fails if it isn't)
cp .env.example .env      # add a free Groq key from console.groq.com
python examples/basic_usage.py
python examples/tool_calling.py
```

CI runs lint, a formatting check, tests with coverage, and a package build on Python 3.10–3.13,
plus one job that installs the oldest dependency versions `pyproject.toml`
allows, so the minimum versions are real.

---

## Known limitations

Listed on purpose, not hidden:

- **No streaming.** Each call returns the full reply.
- **Async only.** There's no synchronous client; use `asyncio.run()` in scripts.
- **Rate limits count requests, not tokens.** Providers also limit tokens per
  minute; Steadfast relies on retrying 429s for that.
- **Tool arguments are parsed, not checked against your schema.** You get a
  dict of valid JSON; check its fields yourself (e.g. `MyArgs(**call.arguments)`).
- **Images and other non-text content** are passed to Anthropic as-is, not
  translated from OpenAI's format.
- **A validator that raises an exception** is treated as a bug and stops the
  call, rather than counting as a failed attempt.
- **Retry-After is capped at 60 seconds**, so a provider asking for longer gets
  retried sooner than it asked.
- **Default model names go stale.** Pass `model=...` for anything that matters.
- **Tested against fakes.** The test suite mocks every SDK; the examples and
  benchmark are what run against a live API.
- **The benchmark is small** and was measured with v0.2 (see above).

---

## Author

**Manish Pandeya** · [github.com/maniesh-lab](https://github.com/maniesh-lab) · MIT License
