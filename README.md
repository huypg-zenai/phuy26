# Image to Three.js miner

Generates a Three.js module for a prompt image through the competition witness: drafts, selection, rewrites against
renders and a final pick, within the per-prompt budget.

Entry point: `main.py` -> `async generate(image, seed, witness) -> str | None`. The plan is in `miner404/versions.py`.

## Live run

```sh
uv venv --python 3.13 .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
git clone --branch miner-runtime-v0.1.5 --depth 1 https://github.com/404-Repo/404-gen-subnet.git ../404-sdk
```

Create `witness.json` next to `main.py`, with `witness_url` set to `https://witness.subnet404.xyz` and
`proof_parameters` set to the absolute path of `404-sdk/witness-server/verifier/params`. Then:

```sh
.venv-live/bin/python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY
```

## Origin

Fork of SmoothTight/newvictor@0f4bc9370c08 (404-GEN round 2), with these changes by huypg-zenai (wren-plus):
- The finished module's outline fit (`__bake404`) moves every baked part directly under the root, so a part nested
  inside another part is no longer offset twice; parts of hidden groups stay hidden, and parts built with
  `matrixAutoUpdate = false` get their matrix updated.
- The second-stage grid that rewrites and selection see is rendered with `lighting=neutral`, as 404's judge renders it
  since 05/11 (shared/subnet_common/render.py:236); the prompts no longer describe a dim grid light.
- Extra material and craft rules in the generation instructions (opacity, near-white, basic materials, surface
  patterns, closed surfaces, food and organic objects).
- Second gate on every keep-1 rewrite step: the panel's pick must also beat its parent in an 8-call duel with 404's
  stage-1 judge prompt on the judge's own front views (both orders x 4 views, Sol low), else the parent stays. The duel
  runs only when its calls fit with every later planned call still left; otherwise the panel decides alone.

## Origins (beat-all L3)

- Base: SmoothTight/newvictor@0f4bc937 (open source), via our wren-plus fork N2 (bake re-parent fix, neutral grid
  previews, sedge material rules, 8-call duel gates).
- beat-all changes: three lineages kept to the end with per-lineage gates (`Step.per_lineage`), no Astra step, final
  panel + 8-call duel between the two best lineages (`StagePlan.final_duel`).
- Optional patch-edit rewrites (`Step.patch`, plan `PATCH_PLAN`, not active): `miner404/patch.py` is copied from
  kaito0311/3d-gen-sub@1b32e25f8564 `worker/patch.py` (MIT); the edit-format text in `prompts.REWRITE_PATCH` is adapted
  from that repo's `PATCH_ASK`.
