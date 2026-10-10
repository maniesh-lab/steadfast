# Steadfast benchmark

Model: `openai/gpt-oss-20b` | 6 tasks x 3 trials per mode | 2026-10-10

| Mode | Requests | Passed | Failed | API errors | Pass rate | Avg API calls | Avg seconds |
|---|---|---|---|---|---|---|---|
| No retry (max_attempts=1) | 18 | 15 | 3 | 0 | 83% | 1.11 | 2.3 |
| With retry (max_attempts=3) | 18 | 18 | 0 | 0 | 100% | 1.56 | 7.8 |

Pass rate excludes API errors. Avg API calls includes transport retries.

Passes per task (out of 3):

| Task | No retry | With retry |
|---|---|---|
| invoice | 3 | 3 |
| verbatim quotes | 3 | 3 |
| review score | 3 | 3 |
| action items | 0 | 3 |
| order total | 3 | 3 |
| contact | 3 | 3 |
