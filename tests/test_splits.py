import pytest

from cyberreason_bench.splits import Case, assign_family_id, family_split, verify_no_leakage, split_manifest


def make_cases(n_families=10, cases_per_family=3):
    cases = []
    for f in range(n_families):
        family_key = assign_family_id(f"source-{f}")
        for c in range(cases_per_family):
            cases.append(Case(id=f"case-{f}-{c}", family_key=family_key, payload={}))
    return cases


def test_family_split_ratios_sum_validation():
    with pytest.raises(ValueError):
        family_split(make_cases(), ratios=(0.5, 0.3, 0.3))


def test_family_split_no_leakage():
    cases = make_cases(n_families=20, cases_per_family=4)
    splits = family_split(cases, ratios=(0.6, 0.2, 0.2))
    leaked = verify_no_leakage(splits)
    assert leaked == []


def test_family_split_covers_all_cases():
    cases = make_cases(n_families=15, cases_per_family=2)
    splits = family_split(cases)
    total = sum(len(v) for v in splits.values())
    assert total == len(cases)


def test_same_source_always_same_family():
    id1 = assign_family_id("CVE-2024-1234")
    id2 = assign_family_id("CVE-2024-1234")
    id3 = assign_family_id("CVE-2024-9999")
    assert id1 == id2
    assert id1 != id3


def test_split_manifest_reports_leakage_and_counts():
    cases = make_cases(n_families=12, cases_per_family=3)
    splits = family_split(cases, ratios=(0.5, 0.25, 0.25))
    manifest = split_manifest(splits)
    assert manifest["n_cases_total"] == len(cases)
    assert manifest["leaked_family_keys"] == []
    assert set(manifest["splits"].keys()) == {"train", "dev", "test"}


def test_verify_no_leakage_detects_deliberate_leak():
    fam = assign_family_id("shared-source")
    splits = {
        "train": [Case(id="a", family_key=fam, payload={})],
        "test": [Case(id="b", family_key=fam, payload={})],
    }
    leaked = verify_no_leakage(splits)
    assert leaked == [fam]
