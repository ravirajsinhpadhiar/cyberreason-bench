"""Bootstrap statistics for small cyber benchmarks. Maps to Index.xlsx item 44.

Goal (from the index): grouped/paired bootstrap at the independent-family
level (not the case level -- see splits.py), case pass rate as the primary
endpoint, and prespecified stop rules, for benchmarks too small for naive
significance testing (a few hundred cases, not the tens of thousands
classical asymptotics assume).

SCAFFOLDING NOTE: the resampling and CI logic here is real and directly
usable. What it can't do for you is pick your actual stop rule (e.g. "we
will not claim system A beats system B unless the paired-bootstrap 95% CI
on the difference in pass rate excludes 0 AND the point estimate exceeds
X") -- that threshold is a methodology decision for the specific paper,
not something this file should silently default for you. `stop_rule_check`
takes it as an explicit argument so it's never implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence: float

    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _family_groups(family_ids: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, fam in enumerate(family_ids):
        groups.setdefault(fam, []).append(i)
    return groups


def grouped_bootstrap_pass_rate(
    passed: Sequence[bool],
    family_ids: Sequence[str],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 13,
) -> BootstrapResult:
    """Bootstrap CI on case pass rate, resampling whole FAMILIES with
    replacement (not individual cases) so within-family correlation
    doesn't produce an artificially tight interval. `n_resamples` defaults
    to 2000 per the index item's spec ('>=2000 resamples')."""
    if len(passed) != len(family_ids):
        raise ValueError("passed and family_ids must be the same length")
    if len(passed) == 0:
        raise ValueError("need at least one case")

    passed_arr = np.asarray(passed, dtype=float)
    groups = _family_groups(family_ids)
    family_keys = list(groups.keys())
    rng = np.random.default_rng(seed)

    point_estimate = float(passed_arr.mean())
    resampled_rates = np.empty(n_resamples)
    for r in range(n_resamples):
        sampled_families = rng.choice(family_keys, size=len(family_keys), replace=True)
        idxs = [i for fam in sampled_families for i in groups[fam]]
        resampled_rates[r] = passed_arr[idxs].mean()

    alpha = 1 - confidence
    lo = float(np.quantile(resampled_rates, alpha / 2))
    hi = float(np.quantile(resampled_rates, 1 - alpha / 2))
    return BootstrapResult(point_estimate, lo, hi, n_resamples, confidence)


def paired_bootstrap_diff(
    passed_a: Sequence[bool],
    passed_b: Sequence[bool],
    family_ids: Sequence[str],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 13,
) -> BootstrapResult:
    """Bootstrap CI on the PAIRED difference in pass rate (system A - system
    B) on the same cases, resampled at the family level. Use this to test
    whether one system beats another, not two independent grouped_bootstrap
    calls compared by eye."""
    if not (len(passed_a) == len(passed_b) == len(family_ids)):
        raise ValueError("passed_a, passed_b, and family_ids must be the same length")

    a = np.asarray(passed_a, dtype=float)
    b = np.asarray(passed_b, dtype=float)
    diff = a - b
    groups = _family_groups(family_ids)
    family_keys = list(groups.keys())
    rng = np.random.default_rng(seed)

    point_estimate = float(diff.mean())
    resampled_diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        sampled_families = rng.choice(family_keys, size=len(family_keys), replace=True)
        idxs = [i for fam in sampled_families for i in groups[fam]]
        resampled_diffs[r] = diff[idxs].mean()

    alpha = 1 - confidence
    lo = float(np.quantile(resampled_diffs, alpha / 2))
    hi = float(np.quantile(resampled_diffs, 1 - alpha / 2))
    return BootstrapResult(point_estimate, lo, hi, n_resamples, confidence)


def stop_rule_check(result: BootstrapResult, min_point_estimate: float) -> bool:
    """Explicit, no-default stop-rule evaluator: only claim a win if the
    CI excludes zero AND the point estimate clears a threshold you name
    ahead of time (item 44/24's prespecified-stop-rule requirement)."""
    return result.excludes_zero() and result.point_estimate >= min_point_estimate
