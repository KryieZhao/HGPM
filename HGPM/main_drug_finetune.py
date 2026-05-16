"""HGPM drug-side regime / adverse-event finetuning entrypoint.

Wraps :func:`HGPM.task.drug.semantic_finetune_hgpm.main` so the finetune
driver can be invoked with ``python -m HGPM.main_drug_finetune``.
"""

from __future__ import annotations

from HGPM.task.drug.semantic_finetune_hgpm import main


if __name__ == "__main__":
    main()
