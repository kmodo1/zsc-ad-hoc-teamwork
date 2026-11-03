"""Full evaluation table: every method against every held-out partner, in both
player positions (Overcooked layouts are not symmetric, so one position alone
is not a fair measure).

Results are appended to a JSONL file one cell at a time and finished cells are
skipped on restart, so a preempted Slurm job loses at most one cell of work.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from adhoc_teamwork.envs.vec_env import VecEnv
from adhoc_teamwork.evaluation.rollout import run_episodes_batched
from adhoc_teamwork.policies.agent_api import Agent


def _load_done(path: str) -> Dict[Tuple[str, str, int], dict]:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line truncated by preemption; that cell gets redone
                done[(row["method"], row["partner"], row["position"])] = row
    return done


def run_benchmark(
    vec_env: VecEnv,
    methods: Dict[str, Callable[[], Agent]],
    partners: Dict[str, Callable[[], Agent]],
    out_path: str,
    seed: int = 0,
    verbose: bool = True,
) -> List[dict]:
    """Returns one row per (method, partner, position). The same episode seed
    is used for every method on a given (partner, position), so comparisons
    between methods share partner sampling noise as far as possible.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _load_done(out_path)
    partner_seed = {name: seed + 1000 * i for i, name in enumerate(partners)}
    with open(out_path, "a") as f:
        # After a line truncated by preemption, start a new line; otherwise the
        # next row would be glued onto the fragment and lost as well.
        if f.tell() > 0:
            with open(out_path, "rb") as existing:
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) != b"\n":
                    f.write("\n")
        for method, make_method in methods.items():
            for partner, make_partner in partners.items():
                for position in (0, 1):
                    if (method, partner, position) in done:
                        continue
                    pair = [make_method(), make_partner()]
                    if position == 1:
                        pair.reverse()
                    result = run_episodes_batched(vec_env, pair, seed=partner_seed[partner] + position)
                    row = {
                        "method": method,
                        "partner": partner,
                        "position": position,
                        "returns": [float(r) for r in result.returns],
                    }
                    done[(method, partner, position)] = row
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            if verbose:
                cells = [r for k, r in done.items() if k[0] == method]
                print(f"[benchmark] {method}: mean return {np.mean([np.mean(r['returns']) for r in cells]):.1f} over {len(cells)} cells", flush=True)
    return list(done.values())


def method_partner_means(rows: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    """{method: {partner: mean return, averaged over the two positions}}."""
    table: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        table.setdefault(row["method"], {}).setdefault(row["partner"], []).append(float(np.mean(row["returns"])))
    return {m: {p: float(np.mean(v)) for p, v in partners.items()} for m, partners in table.items()}
