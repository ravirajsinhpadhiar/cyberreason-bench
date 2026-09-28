"""Leakage-controlled family-level splits. Maps to Index.xlsx item 42
(and shares its evaluation-protocol discipline with items 19 and 43:
"open-evidence" retrievable relations vs fully "withheld" target relations).

Goal (from the index): split by incident/report family, CVE family, policy
template, or time -- BEFORE paraphrasing -- so no shared-source leakage
crosses train/dev/test. Two documents that are both about the same
incident, the same CVE, or paraphrases of the same policy passage must
never end up split across train and test, or accuracy numbers are
inflated by the model having effectively seen the answer.

SCAFFOLDING NOTE: `family_split` is real, generic splitting logic you can
use as-is. `assign_family_id` ships with a few obvious default rules
(exact CVE id, exact source-document id) -- for a real benchmark you will
need to extend this with your own family-identification logic (e.g.
near-duplicate detection across paraphrased policy passages), which this
scaffold deliberately does not attempt to solve for you.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Case:
    """One evaluable unit (a question/answer pair, a labeled example, etc.)."""

    id: str
    family_key: str  # e.g. CVE id, source-document id, incident id
    payload: dict


def assign_family_id(source_id: str) -> str:
    """Default family assignment: stable hash of the source/document id.

    Two cases sharing the same `source_id` (same CVE, same advisory, same
    policy document) collapse to the same family and can never be split
    across train/dev/test. Replace with near-duplicate/semantic clustering
    if your corpus has paraphrased duplicates without a shared literal id.
    """
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]


def family_split(
    cases: list[Case],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 13,
) -> dict[str, list[Case]]:
    """Split cases into train/dev/test at the FAMILY level, not the case
    level -- every case belonging to one family lands entirely in one
    split. Returns {"train": [...], "dev": [...], "test": [...]}.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    families: dict[str, list[Case]] = {}
    for c in cases:
        families.setdefault(c.family_key, []).append(c)

    family_keys = sorted(families.keys())  # deterministic order before shuffle
    rng = random.Random(seed)
    rng.shuffle(family_keys)

    n = len(family_keys)
    n_train = round(n * ratios[0])
    n_dev = round(n * ratios[1])
    train_keys = set(family_keys[:n_train])
    dev_keys = set(family_keys[n_train:n_train + n_dev])
    test_keys = set(family_keys[n_train + n_dev:])

    out = {"train": [], "dev": [], "test": []}
    for key, bucket in families.items():
        if key in train_keys:
            out["train"].extend(bucket)
        elif key in dev_keys:
            out["dev"].extend(bucket)
        else:
            out["test"].extend(bucket)
    return out


def verify_no_leakage(splits: dict[str, list[Case]]) -> list[str]:
    """Sanity check: no family_key appears in more than one split.
    Returns a list of leaked family_keys (empty list == clean split)."""
    split_of: dict[str, str] = {}
    leaked: set[str] = set()
    for split_name, cases in splits.items():
        for c in cases:
            if c.family_key in split_of and split_of[c.family_key] != split_name:
                leaked.add(c.family_key)
            split_of[c.family_key] = split_name
    return sorted(leaked)


def split_manifest(splits: dict[str, list[Case]]) -> dict:
    """Human-auditable record of a split: which families/cases went where,
    and the leakage check result. Write this alongside any benchmark
    release so the split is independently verifiable."""
    manifest = {
        "n_cases_total": sum(len(v) for v in splits.values()),
        "leaked_family_keys": verify_no_leakage(splits),
        "splits": {},
    }
    for name, cases in splits.items():
        manifest["splits"][name] = {
            "n_cases": len(cases),
            "n_families": len({c.family_key for c in cases}),
            "case_ids": sorted(c.id for c in cases),
        }
    return manifest
