# Data

This directory contains processed data required by DyRCoD. Paths are relative to the project root and configured in `configs/default.yaml`.

## Layout

```text
data/processed/features/
data/processed/networks/
data/processed/labels/
```

## Features

`data/processed/features/` contains:

- `multiomics_features_STRING.tsv`: canonical gene index used by the model.
- `feature_genename.txt`: gene name list aligned with the processed feature/network matrices.
- `mutation/`: cancer-specific mutation subtype features.
- `CNV/`: cancer-specific copy-number variation features.
- `Expression/`: cancer-specific expression features.
- `CRISPR/`: cancer-specific CRISPR dependency features.
- `Spatial/`: spatial or chromatin-derived gene features. Current inputs exclude `chr_id` and keep `relative_pos`, `arm_id`, `dist_centromere_norm`, and `local_gene_density_1mb`.

## Networks

`data/processed/networks/` contains:

- `STRING_ppi.pkl`: STRING PPI network.
- `GO_SimMatrix_filtered_fixed.pkl`: GO semantic similarity network.
- `KEGG_IDF_Cosine_threshold_0.6.pkl`: KEGG/Pathway co-occurrence network.
- `pathway_SimMatrix_filtered_fixed.pkl`: retained pathway similarity network from the old project for compatibility and downstream checks.

## Labels

`data/processed/labels/specific-cancer/` contains cancer-specific labels:

- `label_file-P-<cancer>.txt`: binary label vector.
- `pos-<cancer>.txt`: positive driver gene node indices.
- `pan-neg.txt`: pan-cancer negative gene node indices.

The repository includes processed files only. Raw TCGA downloads, experiment logs, model checkpoints, ablation outputs, and plotting artifacts are intentionally excluded.
