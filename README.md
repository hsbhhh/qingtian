# DyRCoD: Dynamic Rewiring and Role–Community Decoupling for Cancer Driver Gene Prediction

DyRCoD is a dynamic multi-view graph learning model for cancer driver gene prediction. It integrates multi-omics gene features with PPI, GO, and Pathway biological networks, then learns candidate driver gene representations through dynamic directional edge reweighting and role–community decoupling.

## Features

- Multi-omics feature integration.
- Multi-view biological network learning over PPI, GO, and Pathway graphs.
- Dynamic directional edge reweighting for biological network edges.
- Role–community orthogonal decoupling.
- Cancer driver gene prioritization for labeled and unlabeled genes.

## Project Structure

```text
DyRCoD/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   └── processed/
│       ├── features/
│       ├── networks/
│       └── labels/
├── src/
│   └── dyrcod/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── layers.py
│       ├── loss.py
│       ├── metrics.py
│       ├── model.py
│       ├── predict.py
│       ├── train.py
│       └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict_all_genes.py
└── outputs/
    └── .gitkeep
```

`src/dyrcod/` contains the reusable model package. `scripts/` contains command-line entry points. `configs/default.yaml` defines relative paths and model/training parameters. `data/processed/` stores the processed features, biological networks, and cancer labels needed to reproduce training. Runtime artifacts are written to `outputs/`, which is ignored by Git except for `.gitkeep`.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate dyrcod
```

Using pip:

```bash
conda create -n dyrcod python=3.9
conda activate dyrcod
pip install -r requirements.txt
```

DyRCoD depends on PyTorch. Use the official PyTorch install command that matches your CUDA version if GPU training is needed; `torch>=2.0` is recommended.

## Data

Processed data are stored with relative paths:

```text
data/processed/features/
data/processed/networks/
data/processed/labels/
```

`features/` contains the multi-omics feature table, cancer-specific mutation/CNV/expression/CRISPR features, spatial features, and the gene index. `networks/` contains PPI, GO semantic similarity, and KEGG/Pathway network files. `labels/` contains cancer-specific positive driver labels and pan-cancer negative labels.

The default configuration expects the following network views:

- `PPI`: `STRING_ppi.pkl`
- `GO`: `GO_SimMatrix_filtered_fixed.pkl`
- `Pathway`: `KEGG_IDF_Cosine_threshold_0.6.pkl`

## Training

Run 10-fold cross-validation for LUAD:

```bash
python scripts/train.py --config configs/default.yaml --cancer LUAD
```

Common overrides:

```bash
python scripts/train.py --config configs/default.yaml --cancer LUAD --epochs 50 --gpu 0
```

## Evaluation

The evaluation command runs the same stratified cross-validation protocol and writes metrics to `outputs/<CANCER>/`:

```bash
python scripts/evaluate.py --config configs/default.yaml --cancer LUAD
```

Primary output files:

```text
outputs/LUAD/cross_validation_summary.json
outputs/LUAD/cross_validation_metrics.csv
```

## Predict All Genes

Train a final model on all labeled driver and negative genes, then rank all genes:

```bash
python scripts/predict_all_genes.py --config configs/default.yaml --cancer LUAD
```

The default prediction output is:

```text
outputs/LUAD/luad_all_gene_predictions.csv
```

## Outputs

All runtime outputs are written under:

```text
outputs/
```

This directory is ignored by Git so that logs, prediction CSVs, checkpoints, and intermediate artifacts are not committed accidentally.

## Citation

If you use DyRCoD in your research, please cite our manuscript:

```bibtex
@article{hu2026dyrcod,
  title={DyRCoD: Dynamic Rewiring and Role--Community Decoupling for Cancer Driver Gene Prediction},
  author={Hu, Shaobo and Zhao, Ning and Zhang, Chunlong},
  journal={Briefings in Bioinformatics},
  year={2026}
}
```

The citation information will be updated after publication.

## License

This project is released under the MIT License.
