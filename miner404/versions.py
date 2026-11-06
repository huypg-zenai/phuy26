"""The plan this submission runs (beat-all L3P5 = L3 + a 5th patch-edit step; L3: three lineages to the end, per-lineage gates, final panel + 8-call duel)."""

from __future__ import annotations

from .pipeline import Draft, StagePlan, Step

SOL_HIGH = ("gpt-6.1-sol", "high")
ACTIVE_PLAN = StagePlan(
    drafts=tuple(Draft(start=start, model="gpt-6.1-sol", effort="high") for start in ("plain", "proportions") * 3),
    draft_keep=3,
    # beat-all L3P5: L3's 4 full Sol-high rewrite steps + a 5th PATCH-EDIT step per lineage (T-patch: an edit is as good as a
    # full rewrite, ~27% cheaper; the L3 live smoke used only 2.6-3.4M of 5M credits), every step gated within its lineage.
    steps=(*(Step(rewrites=(SOL_HIGH,), per_lineage=True) for _ in range(4)),
           Step(rewrites=(SOL_HIGH,), per_lineage=True, patch=True)),
    pose=True, fill=0.99, glow=0.0, measured=False, outline_fit=False, outline_fit_last=True,
    final_duel=True,
)

# beat-all L3P candidate (idea I3, not active): two full rewrites per lineage, then three cheaper patch-edit steps
# (SEARCH/REPLACE on the parent's module), every step gated within its lineage. Activate only if T-patch passes.
PATCH_PLAN = StagePlan(
    drafts=ACTIVE_PLAN.drafts,
    draft_keep=3,
    steps=(*(Step(rewrites=(SOL_HIGH,), per_lineage=True) for _ in range(2)),
           *(Step(rewrites=(SOL_HIGH,), per_lineage=True, patch=True) for _ in range(3))),
    pose=True, fill=0.99, glow=0.0, measured=False, outline_fit=False, outline_fit_last=True,
    final_duel=True,
)
