"""HGPM graph-side node-classification finetuning entrypoint.

Wraps :func:`HGPM.task.graph.semantic_finetune_hgpm.main` so the finetune
driver can be invoked with ``python -m HGPM.main_graph_finetune``.
"""

from __future__ import annotations

from HGPM.task.graph.semantic_finetune_hgpm import main


if __name__ == "__main__":
    main()
