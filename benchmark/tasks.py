"""
Benchmark tasks for Steadfast.

Each task is: some messy text, a Pydantic schema to extract it into, and
validators that check rules the schema can't express (verbatim quotes,
date arithmetic, sums that must add up).

Validators describe the RULE that was broken. They never reveal the
correct answer, otherwise the retry would be trivially easy.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class Task:
    name: str
    instruction: str
    source: str
    schema: type[BaseModel]
    validators: list[Callable] = field(default_factory=list)


def _iso(value):
    """Parse YYYY-MM-DD into a date, or None if it isn't a real date."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------- 1. invoice

INVOICE_TEXT = """INVOICE #INV-2025-0417
Himalayan Roast Co., Pokhara
Issued: 12 March 2025   Payment due within 14 days.
Item: Green coffee beans (20 kg) .......... NPR 38,000.00
Shipping ................................... NPR 2,000.00
TOTAL DUE: NPR 40,000.00"""


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    total: float
    currency: Literal["USD", "EUR", "GBP", "NPR"]
    due_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


def _invoice_number_in_text(r):
    if r.invoice_number in INVOICE_TEXT:
        return None
    return "invoice_number must appear verbatim in the text"


def _due_date_is_14_days_after_issue(r):
    d = _iso(r.due_date)
    if d is None:
        return "due_date is not a real calendar date"
    if d == date(2025, 3, 12) + timedelta(days=14):
        return None
    return "due_date must be exactly 14 days after the issue date"


def _total_matches_total_due(r):
    if abs(r.total - 40000.0) < 0.01:
        return None
    return "total must equal the TOTAL DUE amount in the text"


# --------------------------------------------------------- 2. verbatim quotes

NEWS_TEXT = (
    "The Pokhara municipal council voted on Tuesday to ban single-use plastic "
    "bags in the lakeside market area starting in January. Council member Anita "
    "Gurung said the ban was overdue, noting that plastic waste now clogs three "
    "of the five drainage channels feeding Phewa Lake. Shopkeepers will receive a "
    "two-month grace period and subsidised cloth bags. Critics, including the "
    "Lakeside Traders Association, argued that enforcement would fall on small "
    "vendors who cannot afford fines, and asked the council to delay the ban "
    "until June. The council said the January date would not move, but promised "
    "to review fines after six months."
)


class Claim(BaseModel):
    claim: str
    quote: str


class Evidence(BaseModel):
    claims: list[Claim] = Field(min_length=3, max_length=3)


def _quotes_are_verbatim(r):
    bad = [str(i + 1) for i, c in enumerate(r.claims) if c.quote.strip() not in NEWS_TEXT]
    if not bad:
        return None
    return f"quote is not a verbatim substring of the text, in claim(s): {', '.join(bad)}"


# ------------------------------------------------------------ 3. review score


REVIEW_TEXT = (
    "I bought this blender three weeks ago. The motor is powerful and it crushes "
    "ice in seconds, but the lid leaks whenever I fill it past the halfway line, "
    "and the jug is a pain to clean. Customer support replaced the lid quickly, "
    "which I appreciated. Overall it's decent for the price, though I'd hesitate "
    "to recommend it to anyone who wants to make big batches."
)


class ReviewSummary(BaseModel):
    score: int = Field(ge=1, le=5)
    sentiment: Literal["positive", "negative", "mixed"]
    aspects: list[str] = Field(min_length=3, max_length=3)


def _sentiment_matches_score(r):
    expected = "negative" if r.score <= 2 else "mixed" if r.score == 3 else "positive"
    if r.sentiment == expected:
        return None
    return "sentiment must be consistent with score (1-2 negative, 3 mixed, 4-5 positive)"


# ------------------------------------------------------------ 4. action items

MEETING_TEXT = """Meeting date: Monday 6 October 2025. Attendees: Sunita, Rajan, Mei.
- Sunita will send the revised budget to finance by Friday.
- Rajan to book the venue for the November workshop before the end of next week.
- Mei will draft the invitation text and share it by Wednesday 15 October."""


class ActionItem(BaseModel):
    owner: str
    task: str
    deadline: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class ActionList(BaseModel):
    items: list[ActionItem] = Field(min_length=1)


def _owners_in_text(r):
    bad = [i.owner for i in r.items if i.owner not in MEETING_TEXT]
    if not bad:
        return None
    return "every owner must be a name that appears in the text"


def _deadlines_are_sensible(r):
    meeting = date(2025, 10, 6)
    for item in r.items:
        d = _iso(item.deadline)
        if d is None:
            return "every deadline must be a real calendar date"
        if not meeting < d <= meeting + timedelta(days=30):
            return "every deadline must fall after the meeting date and within 30 days of it"
        if d.weekday() >= 5:
            return "every deadline in the text is a working day, but one of yours is a weekend"
    return None


# ------------------------------------------------------------------ 5. order

ORDER_TEXT = (
    "Order #88: 2x notebook @ 3.50 each; 5 pens @ 0.80 each; 1 stapler @ 12.25. "
    "Customer asked for the order total."
)


class OrderItem(BaseModel):
    name: str
    qty: int
    unit_price: float


class Order(BaseModel):
    items: list[OrderItem]
    total: float


def _total_is_sum_of_items(r):
    expected = sum(i.qty * i.unit_price for i in r.items)
    if abs(r.total - expected) < 0.01:
        return None
    return "total must equal the sum of qty x unit_price across all items"


def _item_names_in_text(r):
    bad = [i.name for i in r.items if i.name.lower() not in ORDER_TEXT.lower()]
    if not bad:
        return None
    return "every item name must appear in the text exactly as written there"


# ---------------------------------------------------------------- 6. contact

CONTACT_TEXT = (
    "Thanks again for yesterday! You can reach me any time \u2014 Priya Sharma, "
    "Sharma Designs | priya.s@sharmadesigns.example | +977 98-1234-5678 (Viber/WhatsApp)"
)


class Contact(BaseModel):
    name: str
    email: str
    phone: str = Field(pattern=r"^\+?\d{7,15}$")


def _contact_fields_in_text(r):
    if r.email in CONTACT_TEXT and r.name in CONTACT_TEXT:
        return None
    return "name and email must appear verbatim in the text"


# ----------------------------------------------------------------- the list

TASKS = [
    Task(
        "invoice",
        "Extract the invoice details from this text. Dates as YYYY-MM-DD.",
        INVOICE_TEXT,
        Invoice,
        [_invoice_number_in_text, _due_date_is_14_days_after_issue, _total_matches_total_due],
    ),
    Task(
        "verbatim quotes",
        "Pick the three most important claims in this article, each with a supporting quote copied exactly from the text.",
        NEWS_TEXT,
        Evidence,
        [_quotes_are_verbatim],
    ),
    Task(
        "review score",
        "Summarise this product review: a 1-5 score, the overall sentiment, and exactly three aspects it mentions.",
        REVIEW_TEXT,
        ReviewSummary,
        [_sentiment_matches_score],
    ),
    Task(
        "action items",
        "List every action item from these meeting notes with its owner and deadline. Dates as YYYY-MM-DD.",
        MEETING_TEXT,
        ActionList,
        [_owners_in_text, _deadlines_are_sensible],
    ),
    Task(
        "order total",
        "Extract every line item from this order and compute the order total.",
        ORDER_TEXT,
        Order,
        [_total_is_sum_of_items, _item_names_in_text],
    ),
    Task(
        "contact",
        "Extract the person's contact details. The phone number must be digits only, with an optional leading +.",
        CONTACT_TEXT,
        Contact,
        [_contact_fields_in_text],
    ),
]
