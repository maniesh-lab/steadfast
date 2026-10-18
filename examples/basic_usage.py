"""
Basic usage: a plain call, a schema-validated call, and token usage.

Run from the repo root:
    pip install -e ".[providers,dev]"
    python examples/basic_usage.py

Needs GROQ_API_KEY in your environment or a .env file at the repo root.
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel

from steadfast import SteadfastClient, configure_logging
from steadfast.providers.groq_provider import GroqProvider

load_dotenv()
configure_logging()


class Fact(BaseModel):
    topic: str
    fact: str


async def main():
    client = SteadfastClient()
    client.register(GroqProvider(), rate=30, period=60)

    # Plain call: just get text back
    answer = await client.complete("groq", "Say hello in exactly one word.")
    print("Plain response:", answer)

    # Schema-validated call: retries automatically if the shape is wrong
    fact = await client.complete(
        "groq",
        "Give me one interesting fact about octopuses. "
        'Respond ONLY as JSON matching this shape: {"topic": "...", "fact": "..."}',
        schema=Fact,
        system="You are a precise assistant that answers in JSON.",
    )
    print("Validated response:", fact)

    print("Tokens used:", client.total_usage.total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
