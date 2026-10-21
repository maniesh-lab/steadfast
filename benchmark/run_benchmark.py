"""
Benchmark: does validate-and-retry actually help?

Runs every task in tasks.py twice with the same validators:
  - no retry:   max_attempts=1  (one shot, like calling the model directly)
  - with retry: max_attempts=3  (Steadfast feeds the problems back)

Run from the repo root:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --trials 5 --rate 10 --model <groq-model>

Needs GROQ_API_KEY in your environment or .env. Writes benchmark/results.md.
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from tasks import TASKS

from steadfast.client import SteadfastClient
from steadfast.exceptions import ProviderError, ValidationRetryExceeded
from steadfast.logging_utils import configure_logging
from steadfast.providers.base import BaseProvider

MAX_TOKENS = 2048  # headroom: reasoning models spend tokens before answering


class CountingProvider(BaseProvider):
    """Wraps a real provider and counts every API call made through it
    (including transport retries), so we can report the cost of retrying."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.calls = 0

    async def chat(self, messages, tools=None, tool_choice=None, **params):
        self.calls += 1
        return await self.inner.chat(messages, tools=tools, tool_choice=tool_choice, **params)


def build_prompt(task):
    schema_json = json.dumps(task.schema.model_json_schema())
    return (
        f"{task.instruction}\n\n"
        f"Return JSON matching this schema:\n{schema_json}\n\n"
        f"TEXT:\n{task.source}"
    )


async def run_arm(client, provider, label, max_attempts, trials):
    """Run every task `trials` times. Returns one dict per request."""
    print(f"\n== {label} (max_attempts={max_attempts}) ==")
    rows = []
    for task in TASKS:
        for _ in range(trials):
            calls_before = provider.calls
            start = time.perf_counter()
            try:
                await client.complete(
                    provider.name,
                    build_prompt(task),
                    schema=task.schema,
                    validators=task.validators,
                    max_attempts=max_attempts,
                    max_tokens=MAX_TOKENS,
                )
                outcome = "pass"
            except ValidationRetryExceeded:
                outcome = "fail"
            except ProviderError:
                outcome = "error"  # API trouble, not the model's fault

            row = {
                "task": task.name,
                "outcome": outcome,
                "calls": provider.calls - calls_before,
                "seconds": time.perf_counter() - start,
            }
            rows.append(row)
            print(f"  {outcome:5} {task.name:16} calls={row['calls']} {row['seconds']:.1f}s")
    return rows


def summarize(rows):
    passed = sum(r["outcome"] == "pass" for r in rows)
    failed = sum(r["outcome"] == "fail" for r in rows)
    errors = sum(r["outcome"] == "error" for r in rows)
    scored = passed + failed  # errors are excluded from the pass rate
    return {
        "requests": len(rows),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / scored if scored else 0.0,
        "avg_calls": sum(r["calls"] for r in rows) / len(rows),
        "avg_seconds": sum(r["seconds"] for r in rows) / len(rows),
    }


def per_task_passes(rows, trials):
    return {
        task.name: sum(r["outcome"] == "pass" for r in rows if r["task"] == task.name)
        for task in TASKS
    }


def render_markdown(model, trials, base_rows, retry_rows):
    base, retry = summarize(base_rows), summarize(retry_rows)
    base_pt, retry_pt = per_task_passes(base_rows, trials), per_task_passes(retry_rows, trials)

    def row(label, s):
        return (
            f"| {label} | {s['requests']} | {s['passed']} | {s['failed']} | {s['errors']} "
            f"| {s['pass_rate']:.0%} | {s['avg_calls']:.2f} | {s['avg_seconds']:.1f} |"
        )

    lines = [
        "# Steadfast benchmark",
        "",
        f"Model: `{model}` | {len(TASKS)} tasks x {trials} trials per mode | {date.today()}",
        "",
        "| Mode | Requests | Passed | Failed | API errors | Pass rate | Avg API calls | Avg seconds |",
        "|---|---|---|---|---|---|---|---|",
        row("No retry (max_attempts=1)", base),
        row("With retry (max_attempts=3)", retry),
        "",
        "Pass rate excludes API errors. Avg API calls includes transport retries.",
        "",
        f"Passes per task (out of {trials}):",
        "",
        "| Task | No retry | With retry |",
        "|---|---|---|",
    ]
    for task in TASKS:
        lines.append(f"| {task.name} | {base_pt[task.name]} | {retry_pt[task.name]} |")
    return "\n".join(lines) + "\n"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3, help="runs per task per mode")
    parser.add_argument("--rate", type=int, default=12, help="max API calls per minute")
    parser.add_argument("--model", default=None, help="Groq model name (default: provider default)")
    args = parser.parse_args()

    load_dotenv()
    configure_logging(logging.ERROR)

    from steadfast.providers.groq_provider import GroqProvider

    inner = GroqProvider(model=args.model) if args.model else GroqProvider()
    provider = CountingProvider(inner)
    client = SteadfastClient()
    client.register(provider, rate=args.rate, period=60)

    base_rows = await run_arm(client, provider, "No retry", 1, args.trials)
    retry_rows = await run_arm(client, provider, "With retry", 3, args.trials)

    report = render_markdown(inner.model, args.trials, base_rows, retry_rows)
    out = Path(__file__).parent / "results.md"
    out.write_text(report)
    print("\n" + report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
