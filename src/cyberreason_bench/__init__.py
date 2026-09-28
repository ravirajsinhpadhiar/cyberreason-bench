"""cyberreason_bench: evaluation harness scaffolding for the Cyber Reason LM programme.

Modules map to numbered items in the researcher's Index.xlsx research backlog:
  - schema.py    -> item 2  (provenance-aware cyber graph schema)
  - retrieval.py -> item 4  (exact-ID lexical + dense hybrid retrieval)
  - splits.py    -> item 42 (leakage-controlled family-level splits)
  - stats.py     -> item 44 (bootstrap statistics for small benchmarks)

Every module documents, in its own docstring, exactly which parts are
production-shaped and which are stand-ins that must be swapped before
this is used for a real submission (see each module's "SCAFFOLDING NOTE").
"""

from . import schema, retrieval, splits, stats

__all__ = ["schema", "retrieval", "splits", "stats"]
