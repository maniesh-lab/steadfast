"""
Tool calling: a tiny agent loop.

The model can call one tool, get_order_status. We run the tool when it
asks, send the result back, and repeat until it answers in plain text.
This is the loop an agent framework like LangGraph runs for you.

Run from the repo root:
    pip install -e ".[providers,dev]"
    python examples/tool_calling.py

Needs GROQ_API_KEY in your environment or a .env file at the repo root.
"""

import asyncio
import json

from dotenv import load_dotenv
from pydantic import BaseModel

from steadfast import SteadfastClient, function_tool, tool_result
from steadfast.providers.groq_provider import GroqProvider

load_dotenv()

# A fake "database" so the example runs on its own.
ORDERS = {"A100": "shipped on 12 Oct", "A101": "waiting for payment"}


class OrderStatusArgs(BaseModel):
    order_id: str


def get_order_status(order_id: str) -> str:
    return ORDERS.get(order_id, "no order with that ID")


TOOLS = [
    function_tool(
        "get_order_status",
        "Look up the current status of an order by its ID, e.g. A100.",
        OrderStatusArgs,
    )
]

MAX_STEPS = 5  # never let the loop run forever


async def main():
    client = SteadfastClient()
    client.register(GroqProvider(), rate=30, period=60)

    messages = [
        {"role": "system", "content": "Answer using the tools. Be brief."},
        {"role": "user", "content": "What's the status of orders A100 and A101?"},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = await client.chat("groq", messages, tools=TOOLS)

        if not response.tool_calls:
            print("Answer:", response.text)
            break

        # Keep the model's tool request in the conversation, then add
        # one result message per tool call.
        messages.append(response.to_message())
        for call in response.tool_calls:
            args = OrderStatusArgs(**call.arguments)
            result = get_order_status(args.order_id)
            print(f"step {step}: {call.name}({json.dumps(call.arguments)}) -> {result}")
            messages.append(tool_result(call, result))
    else:
        print(f"Stopped after {MAX_STEPS} steps without a final answer.")

    print("Tokens used:", client.total_usage.total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
