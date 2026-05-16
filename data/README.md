# Datasets

This folder is the drop-in root for HGPM datasets. The repository ships it
empty; you fill it by downloading the released archives.

## Download

Get both archives from the
[HGPM dataset folder on Google Drive](https://drive.google.com/drive/folders/1p6MTkg96RaH4wA0w9bVExHoyGWr-8AJJ?usp=drive_link):

| File                          | SHA256                                                             |
|-------------------------------|--------------------------------------------------------------------|
| `hgpm_graph_data_package.zip` | `E5AE5124A88C4E56418217AB29C5C8FC48E332BD183F8DBF6EADF3EB97312342` |
| `hgpm_drug_data_package.zip`  | `02D62F0692408F157A5390F1D67A54417FB58170B2DCDC0A3E8EC515A01734E5` |

Verify (optional):

```bash
sha256sum hgpm_graph_data_package.zip hgpm_drug_data_package.zip
```

## Unpack

Unzip both archives at the **repository root** (one level above this folder)
so the layout becomes:

```text
data/
├── graph/
│   ├── protocols/
│   └── dags/
└── drug/
    ├── hoddi/
    └── jader/
```

```bash
# from the repository root
unzip hgpm_graph_data_package.zip
unzip hgpm_drug_data_package.zip
```

## Alternative: rebuild from local sources

If you already have the original local workspace assets, you can restage them
into the package layout with:

```bash
python data/prepare_graph_data.py
python data/prepare_drug_data.py
```

The contents under `graph/{protocols,dags}/` and `drug/{hoddi,jader}/` are
`.gitignore`d so they will not be committed.
