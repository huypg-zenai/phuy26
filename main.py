"""Miner entry point: the competition runtime supplies one prompt image, the round seed and the witness client.

Live runs: python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from miner404.pipeline import run
from miner404.versions import ACTIVE_PLAN

__all__ = ["generate"]


class _Bridge:
    """Blocking create() and render() over the async witness, for the pipeline's worker threads."""

    def __init__(self, witness: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._witness = witness
        self._loop = loop

    def _wait(self, call: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(call, self._loop).result()

    def create(self, *, request_id: int = 0, **params: Any) -> Any:
        return self._wait(self._witness.responses.create(request_id=request_id, **params))

    def render(self, source: str, options: Any) -> bytes:
        return self._wait(self._witness.render(source, options))


async def generate(image: bytes, seed: int, witness: Any) -> str | None:
    bridge = _Bridge(witness, asyncio.get_running_loop())
    return await asyncio.to_thread(run, image, seed, bridge, bridge, ACTIVE_PLAN)


async def main() -> None:
    from miner_runtime.batch import run as run_batch  # live runner only; generation and replay never import it

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompts", type=Path, help="text file with one image URL per line")
    inputs.add_argument("--image", action="append", help="image file or HTTP(S) URL; repeat for a small batch")
    parser.add_argument("--seed", type=int, required=True, help="published round seed")
    parser.add_argument("--parallel-prompts", type=int, default=12)
    parser.add_argument("--parallel-calls", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=10, help="passes over unfinished prompts; a missing prompt loses")
    parser.add_argument("--wallet-name", default="miner")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets")
    parser.add_argument("--api", action="store_true", help="use OPENAI_API_KEY instead of the local Codex login")
    args = parser.parse_args()

    sources = args.image or [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    await run_batch(
        generate,
        {Path(source).stem: source for source in sources},
        args.seed,
        witness_config=Path(__file__).resolve().with_name("witness.json"),
        workers=args.parallel_prompts,
        parallel_calls=args.parallel_calls,
        attempts=args.attempts,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
        api=args.api,
    )


if __name__ == "__main__":
    asyncio.run(main())
