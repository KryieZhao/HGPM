"""Hypergraph Pattern Machine (HGPM).

Inclusion-aware Transformer for higher-order interactions. The package is
organized as:

- :mod:`HGPM.data`  — graph / drug dataset builders, collators, loaders
- :mod:`HGPM.model` — paper-aligned encoders under ``model/{graph,drug}/hgpm.py``
- :mod:`HGPM.task`  — pretraining, finetuning, and evaluation drivers
- :mod:`HGPM.utils` — io, logging, metrics, seeding, visualization

The ``main_*`` modules at the package root are thin entrypoints that delegate
to the corresponding task driver.
"""

__version__ = "1.0.0"
