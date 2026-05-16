"""HGPM drug-side masked-semantic pretraining entrypoint.

Wraps :func:`HGPM.task.drug.semantic_pretrain_hgpm.main` so the pretraining
driver can be invoked with ``python -m HGPM.main_drug_pretrain``.
"""

from __future__ import annotations

from HGPM.task.drug.semantic_pretrain_hgpm import main


if __name__ == "__main__":
    main()
