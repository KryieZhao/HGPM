"""HGPM graph-side masked-semantic pretraining entrypoint.

Wraps :func:`HGPM.task.graph.semantic_pretrain_hgpm.main` so the pretraining
driver can be invoked with ``python -m HGPM.main_graph_pretrain``.
"""

from __future__ import annotations

from HGPM.task.graph.semantic_pretrain_hgpm import main


if __name__ == "__main__":
    main()
