# Beyond Polarity: Sentiment Analysis of IMDb Movie Reviews

Final project for **NLP and Text Analysis** — Data Science for Society and Business.
Author: Atul Garg, Constructor University.

## Overview

This project classifies IMDb movie reviews as positive or negative, then goes
beyond the standard benchmark with four diagnostic analyses:

1. **Contextual model comparison** — a pre-trained DistilBERT classifier is run
   on the same held-out sample as the linear baselines, and the disagreements
   between the two model families are examined directly.
2. **Adversarial probes** — a hand-crafted suite of twelve sentences targeting
   negation, contrast, double negation, intensification, implicit sentiment,
   and irony, to show where the bag-of-words assumption breaks.
3. **Label-noise audit** — a sentence-embedding nearest-neighbour search that
   flags training reviews whose neighbours overwhelmingly carry the opposite
   label.
4. **Per-prediction attribution** — SHAP-based token-level explanation of
   individual predictions.

The headline finding: the linear baseline and DistilBERT differ by only one
point on the i.i.d. test sample (88.7% vs 87.7%) but by twenty-five points on
the adversarial probes (58.3% vs 83.3%). Test-set accuracy understates the
qualitative gap between the two model families.

## Files

| File | Description |
|------|-------------|
| `NLP_final_project_enhanced.qmd` | Quarto source document |
| `NLP_final_project_enhanced.ipynb` | Jupyter notebook version |
| `NLP_final_project_enhanced.html` | Rendered HTML report with all outputs |
| `NLP_final_project_report.pdf` | Written report (PDF) |

## How to reproduce

1. Install the dependencies:
   ```
   pip install scikit-learn nltk pandas numpy matplotlib seaborn transformers sentence-transformers shap torch
   ```
2. Download `IMDB Dataset.csv` from Kaggle (link below) and place it in the
   same folder as the `.qmd` file.
3. Render the Quarto document:
   ```
   quarto render NLP_final_project_enhanced.qmd
   ```
   Or open `NLP_final_project_enhanced.ipynb` in Jupyter and run all cells.

The first run downloads about 250 MB of pre-trained model weights
(DistilBERT-SST2 and MiniLM); these are cached locally afterwards.

## Dataset

The IMDb movie review corpus contains 50,000 reviews, balanced 25,000 positive
and 25,000 negative. It is publicly available on Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset file is not included in this repository because it exceeds GitHub's
file size limit. Download it from the link above and place it next to the
notebook before running.
