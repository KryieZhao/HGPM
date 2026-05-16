"""Alias entrypoint for drug-side regime finetuning.

Identical behavior to :mod:`HGPM.main_drug_finetune`; kept so that scripts
that refer to the regime task name continue to work.
"""

from __future__ import annotations

from HGPM.main_drug_finetune import main


if __name__ == "__main__":
    main()
