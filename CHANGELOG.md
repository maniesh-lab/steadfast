# Changelog

## 0.3.0 (unreleased)

### Added
- `SteadfastClient.chat()`: send a list of messages, get back an `LLMResponse`
  (text, tool calls, finish reason, token usage, model, provider) in the same
  shape for every provider.
- Tool calling for OpenAI, Groq, and Anthropic, with `function_tool()`,
  `LLMResponse.to_message()`, and `tool_result()` helpers. Messages and tools
  use OpenAI's format; the Anthropic adapter translates both ways.
- Broken tool-call JSON is fed back to the model and retried, like other bad
  output.
- `fallbacks=[...]`: try other providers when one keeps failing
  (`AllProvidersFailed` if all do).
- `register(..., name=...)`: register the same provider more than once
  (e.g. a big and a small model) under different names.
- A clear error message when a provider's SDK isn't installed.
- Token usage per provider (`client.usage`) and in total (`client.total_usage`).
- Retry settings on the client: `SteadfastClient(max_retries, min_wait, max_wait)`.
- `Retry-After` headers are honored (capped at 60 seconds).
- `timeout=` and any SDK client options on every provider.
- `system=` on `complete()`; async validators.
- Top-level imports: `from steadfast import SteadfastClient, ...`; `__version__`.
- Per-provider install extras: `steadfast-llm[openai]`, `[groq]`, `[anthropic]`.
- `examples/tool_calling.py`.
- CI on every push: lint, tests with coverage, and a package build on
  Python 3.10–3.13, plus a job testing the oldest allowed dependencies.
- SPDX license string in `pyproject.toml`; ruff config.

### Changed
- Validation feedback is sent as conversation turns (the model's reply, then
  the problems) instead of being pasted into one prompt.
- Retry decisions use the HTTP status: 408, 409, 429, and 5xx are retried,
  everything else fails immediately. Groq's `tool_use_failed` is retried.
- All request settings are passed through to the provider (previously only
  `max_tokens`). OpenAI and Groq requests use `max_completion_tokens`.
- A reply cut off by the token limit gets a "be shorter" hint on retry.
- `python-dotenv` is no longer a runtime dependency (only the examples use it).
- Minimum `openai` version is 1.55.3 (1.55.0 crashes with httpx 0.28).

### Fixed
- A correlation ID already bound by the caller was overwritten and then
  removed. It's now reused and restored; `complete()` accepts `correlation_id=`.
- Deprecation warning from `tenacity`'s `initial` parameter.

### Unchanged
- `complete()` keeps its signature and return values. Custom providers that
  only implement `complete(prompt)` still work (without tool calling).

## 0.2.0 (2026-10-10)

- Content-level validate-and-retry: schema failures are fed back to the model.
- Custom validators, tolerant JSON extraction, benchmark.

## 0.1.0

- First release: retries, per-provider rate limiting, Pydantic validation,
  structured logging with correlation IDs, Anthropic/OpenAI/Groq adapters.
