"""Provenance-aware cyber graph schema. Maps to Index.xlsx item 2.

Design goal (from the index): a node/edge schema that distinguishes
*how* a relation came to exist -- explicit (stated verbatim in a source),
deterministic-derived (computed by a rule, e.g. version-range logic),
expert-assessed (a human reviewer's judgment), or model-proposed (an LLM
suggested it and it has not yet been independently confirmed) -- and
carries source/date/reviewer/uncertainty on every edge so nothing in the
graph is trusted more than its provenance warrants.

SCAFFOLDING NOTE: this is a real, usable in-memory schema + store (typed,
validated, serializable). It is NOT wired to a real graph database
(Neo4j/RDFLib/NetworkX backing store) or to a real corpus -- swap
`GraphStore`'s internals for one of those once you have real data volume.
The temporal "as of" query (item 3) is intentionally only stubbed here
with a single `valid_from`/`valid_until` pair per edge; item 3's full
design (event time vs publication time vs ingestion time) is a separate,
larger piece of work and should not be assumed solved by this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class ProvenanceKind(str, Enum):
    """How a relation came to be asserted."""

    EXPLICIT = "explicit"                     # stated verbatim in a source document
    DETERMINISTIC_DERIVED = "deterministic_derived"  # computed by a deterministic rule
    EXPERT_ASSESSED = "expert_assessed"        # a named human reviewer judged it so
    MODEL_PROPOSED = "model_proposed"          # an LLM proposed it; unconfirmed


class NodeType(str, Enum):
    CVE = "cve"
    PRODUCT_VERSION = "product_version"
    ADVISORY = "advisory"
    ATTACK_TECHNIQUE = "attack_technique"
    CONTROL = "control"
    POLICY_PASSAGE = "policy_passage"
    THREAT_ACTOR = "threat_actor"
    MALWARE_FAMILY = "malware_family"
    YARA_RULE = "yara_rule"


class EdgeType(str, Enum):
    AFFECTS = "affects"                # CVE -affects-> product_version
    SUPERSEDES = "supersedes"          # advisory -supersedes-> advisory
    MAPS_TO_TECHNIQUE = "maps_to_technique"   # malware_family/CVE -maps_to-> technique
    SATISFIES_CONTROL = "satisfies_control"   # policy_passage -satisfies-> control
    DETECTS = "detects"                # yara_rule -detects-> malware_family
    ATTRIBUTED_TO = "attributed_to"    # malware_family -attributed_to-> threat_actor
    DERIVED_FROM = "derived_from"      # generic provenance backlink


@dataclass(frozen=True)
class Provenance:
    """Required on every edge. This is the whole point of the schema."""

    kind: ProvenanceKind
    source: str                      # document id / URL / rule name / reviewer note
    asserted_on: date                # when the relation was asserted (not ingested)
    uncertainty: float = 0.0         # 0.0 = certain, 1.0 = maximal uncertainty
    reviewer: Optional[str] = None   # required if kind == EXPERT_ASSESSED
    model_name: Optional[str] = None  # required if kind == MODEL_PROPOSED

    def __post_init__(self):
        if not (0.0 <= self.uncertainty <= 1.0):
            raise ValueError(f"uncertainty must be in [0,1], got {self.uncertainty}")
        if self.kind == ProvenanceKind.EXPERT_ASSESSED and not self.reviewer:
            raise ValueError("expert_assessed provenance requires a reviewer name")
        if self.kind == ProvenanceKind.MODEL_PROPOSED and not self.model_name:
            raise ValueError("model_proposed provenance requires a model_name")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["asserted_on"] = self.asserted_on.isoformat()
        return d


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type.value, "label": self.label, "attrs": self.attrs}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType
    provenance: Provenance
    valid_from: Optional[date] = None   # stub for item-3 temporal support
    valid_until: Optional[date] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def is_valid_as_of(self, as_of: date) -> bool:
        if self.valid_from and as_of < self.valid_from:
            return False
        if self.valid_until and as_of > self.valid_until:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.type.value,
            "provenance": self.provenance.to_dict(),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "attrs": self.attrs,
        }


class GraphStore:
    """Minimal in-memory store. Swap for Neo4j/RDFLib/NetworkX at scale."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes and self._nodes[node.id] != node:
            raise ValueError(f"node id collision with different content: {node.id}")
        self._nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        for endpoint in (edge.src, edge.dst):
            if endpoint not in self._nodes:
                raise ValueError(f"edge references unknown node id: {endpoint}")
        self._edges.append(edge)

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        min_provenance: Optional[set[ProvenanceKind]] = None,
        as_of: Optional[date] = None,
    ) -> list[Edge]:
        """1-hop expansion with optional provenance/time filtering.

        `min_provenance` restricts to edges whose provenance.kind is in the
        given set -- e.g. pass {EXPLICIT, DETERMINISTIC_DERIVED} to exclude
        unconfirmed model-proposed edges from a high-stakes answer path.
        """
        out = []
        for e in self._edges:
            if e.src != node_id and e.dst != node_id:
                continue
            if edge_type is not None and e.type != edge_type:
                continue
            if min_provenance is not None and e.provenance.kind not in min_provenance:
                continue
            if as_of is not None and not e.is_valid_as_of(as_of):
                continue
            out.append(e)
        return out

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for e in self._edges:
            by_kind[e.provenance.kind.value] = by_kind.get(e.provenance.kind.value, 0) + 1
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "edges_by_provenance_kind": by_kind,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges],
            },
            indent=2,
        )
