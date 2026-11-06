# Origin: kaito0311/3d-gen-sub@1b32e25f8564 worker/patch.py (MIT License, Copyright (c) 2026 The authors), copied unchanged
# except this header. beat-all L3 uses it for the optional patch-edit rewrite step (Step.patch).
from __future__ import annotations

import re
from dataclasses import dataclass, field

_BLOCK = re.compile(r"^<{5,9} ?SEARCH[ \t]*\n(.*?)^={5,9}[ \t]*\n(.*?)^>{5,9} ?REPLACE[ \t]*$", re.S | re.M)


@dataclass
class PatchResult:
    code: str | None  # the patched module; None when nothing applied (keep the previous module)
    applied: int = 0
    skipped: list[str] = field(default_factory=list)  # one reason per skipped block
    full: bool = False  # the reply was a whole module, not blocks


def blocks(reply: str) -> list[tuple[str, str]]:
    return [(s, r) for s, r in _BLOCK.findall(reply or "")]


def _rstrip_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _replace_once(code: str, search: str, replace: str) -> tuple[str | None, str]:
    """The code with `search` replaced once, or None and the reason it could not be."""
    if not search.strip():
        return None, "empty SEARCH"
    n = code.count(search)
    if n == 1:
        return code.replace(search, replace, 1), ""
    if n > 1:
        return None, f"SEARCH found {n} times"
    # trailing spaces differ: match on right-stripped lines, then map back to the original line span
    lines, want = code.split("\n"), _rstrip_lines(search).rstrip("\n").split("\n")
    stripped = [line.rstrip() for line in lines]
    hits = [i for i in range(len(lines) - len(want) + 1) if stripped[i:i + len(want)] == want]
    if len(hits) != 1:
        return None, "SEARCH not found" if not hits else f"SEARCH found {len(hits)} times (ignoring trailing spaces)"
    i = hits[0]
    new = replace.rstrip("\n").split("\n") if replace.strip() else []
    return "\n".join(lines[:i] + new + lines[i + len(want):]), ""


def apply(code: str, reply: str, extract_full) -> PatchResult:
    """Apply the reply's blocks to `code`; `extract_full(reply)` returns a whole module from the reply or None."""
    found = blocks(reply)
    if not found:
        full = extract_full(reply)
        return PatchResult(code=full, full=full is not None, skipped=[] if full else ["no blocks and no module"])
    out, result = code, PatchResult(code=None)
    for search, replace in found:
        new, why = _replace_once(out, search, replace)
        if new is None:
            result.skipped.append(why)
        else:
            out, result.applied = new, result.applied + 1
    if result.applied:
        result.code = out
    return result
