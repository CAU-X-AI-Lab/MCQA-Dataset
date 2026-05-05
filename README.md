# MCQA-Dataset

MCQA is a structure-grounded benchmark for evaluating retrieval-augmented generation under multi-constraint question answering.

This repository is the lightweight GitHub entry point for MCQA. It contains documentation, statistics, generation scripts, and small query samples. The full dataset is distributed as compressed archives through GitHub Releases.

## Full Dataset

The complete release contains **283,289 QA pairs**, **138,165 GraphML evidence files**, and **134,294 text evidence files**.

| Dataset | Bridge-Star QA | Multi-Structure QA | Total QA | Total GraphML | Total TXT |
|---|---:|---:|---:|---:|---:|
| CM | 90,000 | 3,871 | 93,871 | 33,871 | 30,000 |
| FB | 172,000 | 4,972 | 176,972 | 90,972 | 90,972 |
| UD | 8,304 | 4,142 | 12,446 | 13,322 | 13,322 |
| **Total** | **270,304** | **12,985** | **283,289** | **138,165** | **134,294** |

## Structural Subsets

MCQA contains the original bridge-star structure plus five additional graph structures.

| Structure | QA Pairs |
|---|---:|
| `bridge_star` | 270,304 |
| `single_edge` | 2,972 |
| `path_4` | 3,000 |
| `star_1hop` | 3,000 |
| `star_2hop` | 2,094 |
| `cycle` | 1,919 |

## Repository Layout

```text
README.md
dataset_card.md
downloads.md
docs/
  schema.md
statistics/
  overall_statistics.csv
  structure_statistics.csv
samples/
  CM/
  FB/
  UD/
scripts/
  generation/
```

The full data archives expand into:

```text
MCQA-Dataset/data/
  CM/
  FB/
  UD/
```

Each subset follows:

```text
queries/queries.csv
evidence/
metadata.json
```

## Download

Download the full dataset from the GitHub Release assets:

| Dataset | Archive | Size |
|---|---|---:|
| CM | `mcqa-cm.zip` | 66.97 MB |
| FB | `mcqa-fb.zip` | 162.87 MB |
| UD | `mcqa-ud.zip` | 95.54 MB |

After downloading, unzip the archives into the repository root or another data directory. Each archive contains paths under `MCQA-Dataset/data/<DATASET>/`.

## Query Schema

All normalized `queries.csv` files use:

```csv
id,dataset,domain,structure,query,answer,evidence_graphml,evidence_text,source_file
```

See [docs/schema.md](docs/schema.md) for details.

## Citation

If you use MCQA, please cite the accompanying paper.
