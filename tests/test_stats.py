import random

import pytest

from cyberreason_bench.stats import (
    grouped_bootstrap_pass_rate, paired_bootstrap_diff, stop_rule_check,
)


def synthetic_families(n_families=30, cases_per_family=4, pass_rate=0.7, seed=1):
    rng = random.Random(seed)
    passed, family_ids = [], []
    for f in range(n_families):
        for c in range(cases_per_family):
            passed.append(rng.random() < pass_rate)
            family_ids.append(f"fam-{f}")
    return passed, family_ids


def test_grouped_bootstrap_point_estimate_matches_mean():
    passed, family_ids = synthetic_families(pass_rate=0.7, seed=1)
    result = grouped_bootstrap_pass_rate(passed, family_ids, n_resamples=500, seed=1)
    assert abs(result.point_estimate - sum(passed) / len(passed)) < 1e-9
    assert 0.0 <= result.ci_low <= result.point_estimate <= result.ci_high <= 1.0


def test_grouped_bootstrap_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        grouped_bootstrap_pass_rate([True, False], ["fam-1"], n_resamples=100)


def test_paired_bootstrap_detects_clear_difference():
    # System A passes ~90%, system B passes ~50% on the same families.
    rng = random.Random(2)
    family_ids, a, b = [], [], []
    for f in range(40):
        for c in range(4):
            family_ids.append(f"fam-{f}")
            a.append(rng.random() < 0.9)
            b.append(rng.random() < 0.5)
    result = paired_bootstrap_diff(a, b, family_ids, n_resamples=1000, seed=2)
    assert result.point_estimate > 0.2
    assert result.excludes_zero()


def test_paired_bootstrap_no_difference_does_not_exclude_zero():
    rng = random.Random(3)
    family_ids, a, b = [], [], []
    for f in range(30):
        for c in range(4):
            family_ids.append(f"fam-{f}")
            outcome = rng.random() < 0.7
            a.append(outcome)
            b.append(outcome)  # identical systems -> diff should hover at 0
    result = paired_bootstrap_diff(a, b, family_ids, n_resamples=500, seed=3)
    assert result.point_estimate == 0.0
    assert not result.excludes_zero()


def test_stop_rule_check_requires_both_conditions():
    passed, family_ids = synthetic_families(pass_rate=0.9, seed=5)
    result = grouped_bootstrap_pass_rate(passed, family_ids, n_resamples=500, seed=5)
    assert stop_rule_check(result, min_point_estimate=0.0) is True
    assert stop_rule_check(result, min_point_estimate=0.99) is False
