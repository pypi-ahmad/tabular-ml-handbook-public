# Tabular ML Handbook: Foundation Models vs. Gradient-Boosted Trees

A hands-on, zero-to-mastery tutorial series comparing **tabular foundation models** (zero-shot,
in-context learning) against **classical gradient-boosted trees** — on real, heavy,
production-grade datasets, with every number in this README pulled directly from an executed
notebook, not a vendor slide.

Eight complete, runnable Jupyter notebooks. Two real datasets under 600,000 rows each. Six
models: three foundation models built specifically for tabular in-context learning (Google TabFM,
TabICLv2, TabPFN v3), one AutoML-native foundation model (Mitra), and three classical GBM
libraries (XGBoost, CatBoost, LightGBM).

> **New here?** Start with [`docs/HANDBOOK.md`](docs/HANDBOOK.md) — the long-form version of this
> README with full theory, code walkthroughs, and references for every model. A comparison-only
> writeup with charts lives in [`docs/COMPARISON.md`](docs/COMPARISON.md) (also as PDF).

## What's in here

| # | Notebook | Model | Task | Dataset | Status |
|---|---|---|---|---|---|
| 1 | `google_tabfm_classification_tutorial.ipynb` | Google TabFM | Binary classification | UCI Adult Income (48,842 rows) | ✅ executed |
| 2 | `google_tabfm_regression_tutorial.ipynb` | Google TabFM | Regression | California Housing (20,640 rows) | ✅ executed |
| 3 | `tabiclv2_regression_tutorial.ipynb` | TabICLv2 | Regression | NYC Green Taxi tips (581,835 rows) | ✅ executed |
| 4 | `catboost_regression_tutorial.ipynb` | CatBoost | Regression | NYC Green Taxi tips (581,835 rows) | ✅ executed |
| 5 | `lightgbm_regression_tutorial.ipynb` | LightGBM | Regression | NYC Green Taxi tips (581,835 rows) | ✅ executed |
| 6 | `tabpfn_v3_classification_tutorial.ipynb` | TabPFN v3 | 7-class classification | Forest Covertype (581,012 rows) | ⚠️ written, not executed — see [TabPFN licensing](#tabpfn-v3-one-time-license-step) |
| 7 | `mitra_classification_tutorial.ipynb` | Mitra | 7-class classification | Forest Covertype (581,012 rows) | ✅ executed |
| 8 | `xgboost_classification_tutorial.ipynb` | XGBoost | 7-class classification | Forest Covertype (581,012 rows) | ✅ executed |

Notebooks 3–5 form one comparable trio (same dataset, same train/test split, same 3,000-row
held-out evaluation set). Notebooks 6–8 form a second comparable trio, likewise. Notebooks 1–2
stand alone — Google TabFM was tutorialized first, on its own pair of classic benchmark datasets,
before this repo grew into a broader model-zoo comparison.

Every notebook is genuinely standalone: its own theory section, its own `uv`-managed environment
setup cell, its own dataset download, its own end-to-end run. You can open any single one without
reading the others.

## Results at a glance

Full breakdowns, plots, and discussion are in [`docs/COMPARISON.md`](docs/COMPARISON.md). Headline
numbers:

**Regression — NYC Green Taxi tip prediction (581,835 rows, same 3,000-row eval set for all three):**

| Model | RMSE ↓ | R² ↑ | Wall-clock | Training data used |
|---|---|---|---|---|
| LinearRegression (baseline) | 2.654 | 0.212 | 5.6s | Full pool |
| TabICLv2 | 2.569 | 0.261 | **7.6s** | 20,000-row context |
| CatBoost | 2.330 | 0.393 | 26.2s | Full pool (462,634 rows) |
| **LightGBM** | **2.261** | **0.428** | **1.9s** | Full pool (462,634 rows) |

**Classification — Forest Covertype, 7-class (581,012 rows, same 3,000-row eval set for all three):**

| Model | Accuracy ↑ | Balanced accuracy ↑ | Wall-clock | Training data used |
|---|---|---|---|---|
| LogisticRegression (baseline) | 0.732 | 0.524 | 29s | Full pool |
| Mitra (zero-shot) | 0.735 | 0.396 | **9.4s** | 4,000-row context |
| Mitra (fine-tuned, 30 steps) | 0.800 | 0.609 | 108.5s | 4,000-row context |
| XGBoost (default) | 0.919 | 0.925 | 247s | Full pool (464,809 rows) |
| **XGBoost (tuned)** | **0.977** | **0.964** | 71s + 55s tuning | Full pool (464,809 rows) |

*TabPFN v3's notebook is written but not executed — see
["TabPFN v3: one-time license step"](#tabpfn-v3-one-time-license-step) below.*

**Google TabFM's own pair of benchmarks:**

| Dataset | Best baseline | TabFM (fast) | TabFM (ensemble) |
|---|---|---|---|
| Adult Income (accuracy / ROC-AUC) | XGBoost: 0.880 / 0.936 | 0.878 / 0.925 | 0.878 / 0.923 |
| California Housing (RMSE / R²) | XGBoost: 0.473 / 0.823 | **0.429 / 0.855** | 0.430 / 0.854 |

The one-line takeaway from all eight notebooks: **a lightly-tuned classical GBM, trained on the
full dataset, is still hard to beat on a genuinely large table** — LightGBM won the regression trio
outright, and a 55-second `RandomizedSearchCV` pass took XGBoost from 91.9% to 97.7% accuracy on
Covertype. Foundation models earn their keep on the *first-pass, zero-tuning* comparison and on
small-to-medium data, not by beating a properly tuned tree ensemble at scale — see
[`docs/COMPARISON.md`](docs/COMPARISON.md) for the full, unhedged discussion.

## Repository structure

```
tabular-ml-handbook/
├── README.md                  <- you are here
├── LICENSE                    <- MIT (covers this repo's code; NOT the models — see below)
├── notebooks/                 <- all 8 .ipynb files, executed in place
├── notebooks_pdf/              <- each notebook rendered to PDF (nbconvert + xelatex)
└── docs/
    ├── HANDBOOK.md             <- long-form: theory, code walkthroughs, references for every model
    ├── HANDBOOK.pdf
    ├── COMPARISON.md           <- dedicated results comparison, charts, discussion
    └── COMPARISON.pdf
```

## Running the notebooks yourself

Every notebook manages its own environment with **[uv](https://docs.astral.sh/uv/)** and
**Python 3.13.13**. The first cell of each notebook installs its own dependencies — you don't need
to pre-install anything except `uv` itself:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv yet
uv venv --python 3.13.13 .venv
uv run --python .venv jupyter lab notebooks/
```

Open any notebook and run all cells top to bottom. Expect the *first* cell of each notebook to
take a few minutes (dependency install + first-time model checkpoint download); everything after
that is fast.

### A note on hardware

Every foundation-model notebook checks for a free GPU and falls back to CPU automatically. Google
TabFM and Mitra run fine on CPU (slower). **TabICLv2** is fast on GPU (single-digit seconds for a
20,000-row context) and still workable on CPU with a smaller context. **TabPFN v3** is the
exception — its own documentation states CPU is only realistic under ~1,000 rows, so a GPU is
close to a requirement for that notebook specifically.

### TabPFN v3: one-time license step

As of this writing, the `tabpfn` package requires a **one-time, interactive license acceptance**
before it will download model weights — this is a change from earlier TabPFN releases and cannot
be automated from a script. To run `tabpfn_v3_classification_tutorial.ipynb` yourself:

1. Open [ux.priorlabs.ai](https://ux.priorlabs.ai) and log in (or register).
2. Accept the license under the "Licenses" tab.
3. Copy your API key from [ux.priorlabs.ai/account](https://ux.priorlabs.ai/account).
4. Before running the notebook: `export TABPFN_TOKEN="<your-api-key>"` (or set it in Python with
   `os.environ["TABPFN_TOKEN"]` before the first `.fit()` call).

The notebook itself is complete and fully written — theory, setup, EDA, zero-shot inference,
interpretation, everything — it just hasn't been run end-to-end in this repo pending that manual
step.

## Model licenses — read before any commercial use

This repository's own code and notebooks are MIT-licensed (see [`LICENSE`](LICENSE)). **The
pretrained model weights each notebook downloads are not** — they carry their own, separate terms
from their respective publishers:

| Model | Weight license | Commercial use |
|---|---|---|
| Google TabFM | TabFM Non-Commercial License v1.0 | ❌ Not permitted |
| TabPFN v3 | Prior Labs License (non-commercial) | ❌ Not permitted |
| TabICLv2 | BSD-3-Clause | ✅ Permitted |
| Mitra | Apache-2.0 | ✅ Permitted |
| XGBoost / CatBoost / LightGBM | Apache-2.0 / Apache-2.0 / MIT | ✅ Permitted |

Verify current terms directly from each publisher before relying on this table for a real decision
— licenses can and do change between releases.

## Why these two datasets

- **NYC Green Taxi (Dec 2016), predicting tip amount** — 581,835 real trip records, two
  high-cardinality categorical columns (pickup/dropoff zone, 233 and 259 distinct values), a
  zero-inflated and right-skewed target, real data-quality problems (negative tip corrections), and
  a genuine leakage trap (`total_amount` includes the target). A realistic regression problem, not
  a cleaned-up textbook one.
- **Forest Covertype, predicting one of 7 cover types** — 581,012 real cartographic records,
  meaningfully imbalanced classes (77x between largest and smallest), and a one-hot-to-categorical
  reconstruction step that gives every model genuine multi-level categorical columns to handle.

Both are large enough that a foundation model's context-size limits and a GBM's full-data training
speed both actually matter — the whole point of this repo.

## Related work

This handbook grew out of an earlier, narrower tutorial pair on Google TabFM alone:
[github.com/pypi-ahmad/google-tabfm-tutorial](https://github.com/pypi-ahmad/google-tabfm-tutorial).
That repo is left as-is; this one supersedes it in scope.

## License

[MIT](LICENSE) for the code and documentation in this repository. See
["Model licenses"](#model-licenses--read-before-any-commercial-use) above for the pretrained
weights each notebook uses.

<p align="center">Made with ❤️ by Ahmad Mujtaba</p>
