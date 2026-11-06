"""Per-prompt call and credit accounting with the round's rate file.

The validator charges every dispatched call: input tokens at the full input weight (cached included) and output
tokens (reasoning included) at the output weight, switching the whole request to long-context weights above
272,000 input tokens. Limits: 60 calls and 5,000,000 credits per prompt. Stages ask `allows()` with a worst-case
estimate before dispatching, so the prompt never exceeds the limits even if every call uses its full output cap.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text())
CREDIT_LIMIT = PRICING["credits_per_prompt"]
CALL_LIMIT = PRICING["calls_per_prompt"]
LONG_CONTEXT = PRICING["long_context_above_input_tokens"]
SAFETY = 0.04  # keep 4% of the credit limit unspent
TOKENS_PER_IMAGE = 3000  # conservative per-image input estimate
CHARS_PER_TOKEN = 3.0  # conservative (code and JSON tokenize densely)


def call_credits(model: str, input_tokens: int, output_tokens: int) -> float:
    weights = PRICING["models"][model]
    long = input_tokens > LONG_CONTEXT
    return (input_tokens * weights["long_input_credits_per_token" if long else "input_credits_per_token"]
            + output_tokens * weights["long_output_credits_per_token" if long else "output_credits_per_token"])


def worst_case(model: str, text_chars: int, images: int, max_output_tokens: int) -> float:
    input_tokens = int(text_chars / CHARS_PER_TOKEN) + images * TOKENS_PER_IMAGE
    return call_credits(model, input_tokens, max_output_tokens)


class Budget:
    """Thread-safe ledger of one prompt's calls. Reservations make concurrent stages safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.credits = 0.0
        self._reserved_calls = 0
        self._reserved_credits = 0.0

    def allows(self, estimates: list[float], keep_calls: int = 0, keep_credits: float = 0.0) -> bool:
        with self._lock:
            calls = self.calls + self._reserved_calls + len(estimates) + keep_calls
            credits = self.credits + self._reserved_credits + sum(estimates) + keep_credits
            return calls <= CALL_LIMIT and credits <= CREDIT_LIMIT * (1 - SAFETY)

    def reserve(self, estimate: float) -> bool:
        with self._lock:
            if (self.calls + self._reserved_calls + 1 > CALL_LIMIT
                    or self.credits + self._reserved_credits + estimate > CREDIT_LIMIT * (1 - SAFETY)):
                return False
            self._reserved_calls += 1
            self._reserved_credits += estimate
            return True

    def settle(self, estimate: float, model: str, usage: object | None) -> None:
        """Release a reservation and charge the actual usage (or the estimate when usage is unknown)."""
        if usage is not None:
            charged = call_credits(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        else:
            charged = estimate
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_credits -= estimate
            self.calls += 1
            self.credits += charged
