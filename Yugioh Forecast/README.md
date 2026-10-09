# Yu-Gi-Oh! Card Price Analysis

An exploratory analysis of Yu-Gi-Oh! trading-card prices, looking at how **rarity** and **printing (set)** affect a card's value, and whether a card's market price can predict its set price.

## Data

`yugioh_enriched.csv` — card listings including name, card type, monster race, set name/code, rarity code, set price, and Cardmarket price.

## Notebooks

### `Yu_GI_OH_Card_Forecast.ipynb` — price by rarity + regression

1. **Cleaning** — standardized rarity codes and defaulted missing rarities to Common.
2. **EDA** — total and average set price grouped by rarity, visualized as a grouped bar chart.
3. **Modeling** — standardized features and fit a **Linear Regression** predicting set price from Cardmarket price, evaluated with **5-fold cross-validation** (MSE, MAE, R² per fold).

**Finding:** R² was approximately 0 across all folds, so Cardmarket price alone has essentially **no linear relationship** with set price. Price is driven by other factors — rarity, set, and collector demand — which points to richer features and non-linear models as the next step.

### `Yu_GI_OH_Scatter_Plot_Exodia.ipynb` — single-card deep dive

Filters to *Exodia the Forbidden One* and plots its price across every set it was printed in, showing how much the same card's value varies by printing.

## How to run

```bash
pip install -r ../requirements.txt
jupyter notebook
```

## Skills demonstrated

Data cleaning · grouped aggregation · data visualization (Matplotlib/Seaborn) · feature scaling · K-Fold cross-validation · regression evaluation
