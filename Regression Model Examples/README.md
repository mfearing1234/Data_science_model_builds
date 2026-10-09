# Regression Model Examples

Reference notebooks covering the main families of linear regression in scikit-learn — when to use each, what problem it solves, and how it behaves.

| Notebook | Technique | When to use it |
|---|---|---|
| `Ordinary_Least_Squares.ipynb` | **OLS** linear regression | Baseline; estimates the conditional mean |
| `Ridge_Regression_Example.ipynb` | **Ridge** (L2 penalty) | Many correlated features; shrinks coefficients to reduce variance |
| `Lasso_regression_example.ipynb` | **Lasso** (L1 penalty) | Feature selection; drives some coefficients to exactly zero |
| `ElasticNet_Reg_example.ipynb` | **ElasticNet** (L1 + L2) | Correlated features where Lasso would arbitrarily pick one; includes regularization-path plots on the diabetes dataset |
| `Quantile_regression_example.ipynb` | **Quantile** regression | Predicting intervals (e.g., 5th–95th percentile) rather than a single point; robust to heavy-tailed noise |
| `Robust_Regression_Examples.ipynb` | **Robust** regression (e.g., RANSAC, Huber, Theil-Sen) | Data with outliers that would distort OLS |

## How to run

```bash
pip install -r ../requirements.txt
jupyter notebook
```
