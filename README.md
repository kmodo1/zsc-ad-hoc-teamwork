# zsc-ad-hoc-teamwork

Implementation of our CMSC673 MARL research proposal: adapting zero-shot
coordination by dynamically re-weighting an ensemble of diverse
coordination policies based on a learned, trajectory-conditioned model of
the current (possibly unseen) partner.

- **`adhoc_teamwork/`** -- our implementation: population training,
  cross-play evaluation, the partner trajectory encoder, the adaptive
  ensemble, baselines, evaluation/ablation, and tests. Start here --
  see [`adhoc_teamwork/README.md`](adhoc_teamwork/README.md) for the full
  writeup and reproduction instructions.
- **`ZSC-Eval/`** -- the vendored benchmark/baseline suite from
  [sjtu-marl/ZSC-Eval](https://github.com/sjtu-marl/ZSC-Eval) that the
  proposal builds on (real Overcooked environment, baseline ZSC
  algorithms). Requires its own Python 3.9 / `gym==0.22` / `numpy==1.22`
  environment -- see `ZSC-Eval/README.md`.
