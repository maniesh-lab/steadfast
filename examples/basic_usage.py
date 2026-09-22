"""
Basic usage example for Steadfast — one plain call, one schema-validated call.

Run with:
    python examples/basic_usage.py

Requires GROQ_API_KEY set in your environment or a .env file at the
repo root.
"""

import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel

from steadfast.client import SteadfastClient
from steadfast.logging_utils import configure_logging
from steadfast.providers.groq_provider import GroqProvider

load_dotenv()
configure_logging()


class Fact(BaseModel):
    topic: str
    fact: str


async def main():
    client = SteadfastClient()
    client.register(GroqProvider(), rate=30, period=60)

    # Plain call — no schema, just get raw text back
    answer = await client.complete("groq", "Say hello in exactly one word.")
    print("Plain response:", answer)

    # Schema-validated call — retries automatically if the shape is wrong
    fact = await client.complete(
        "groq",
        "Give me one interesting fact about octopuses. "
        'Respond ONLY as JSON matching this shape: {"topic": "...", "fact": "..."}',
        schema=Fact,
    )
    print("Validated response:", fact)


if __name__ == "__main__":
    asyncio.run(main())