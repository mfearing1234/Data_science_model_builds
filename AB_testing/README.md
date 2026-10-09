# Marketing A/B Test — Did the Ads Drive Conversions?

Evaluates a marketing experiment in which users were shown either a real **advertisement** (treatment) or a **public service announcement** (control). The goal: determine whether the ad campaign caused a statistically significant increase in conversions, and estimate how many conversions it was responsible for.

## Data

[Marketing A/B Testing (Kaggle)](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) — **588,101 users**:

| Group | Users |
|---|---|
| `ad` (treatment) | 564,577 |
| `psa` (control) | 23,524 |

Fields include whether the user converted, total ads seen, and the day/hour they saw the most ads. Downloaded automatically via `kagglehub`.

## Approach

1. **Data quality checks** — cleaned column names; confirmed no nulls, no duplicate rows, and no user in both groups.
2. **Exposure balance** — compared ads seen per user across groups (means ≈ 24.8 in both) with a **Welch's t-test** to make sure exposure wasn't a confounder.
3. **Frequentist test** — **Chi-square test of independence** on the conversion contingency table.
4. **Bayesian test** — **Beta-Binomial** model with uniform priors; 10,000 posterior draws per group to estimate P(ad > PSA).
5. **Effect size** — 95% confidence interval for the difference in conversion rates and an estimate of conversions attributable to the ads.

## Results

| Metric | Result |
|---|---|
| Chi-square p-value | **≈ 2.0 × 10⁻¹³** — reject the null hypothesis |
| Bayesian P(ad conversion rate > PSA) | **≈ 100%** |
| 95% CI for conversion-rate lift | **+0.60 to +0.94 percentage points** |
| Conversions attributable to ads | **≈ 4,343** |

**Conclusion:** the ad campaign produced a real, statistically significant lift in conversions. Frequentist and Bayesian methods agree.

## How to run

```bash
pip install -r ../requirements.txt
jupyter notebook Marketing_ab_testing.ipynb
```

## Skills demonstrated

Experiment design review · hypothesis testing · Chi-square · Welch's t-test · Bayesian inference · confidence intervals · business impact estimation
