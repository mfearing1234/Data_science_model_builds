# Feature Transformations with Ensembles of Trees

Explores using tree ensembles not just as classifiers, but as **feature transformers**: each sample is mapped to the leaves it lands in across the trees, those leaf indices are one-hot encoded into a high-dimensional sparse representation, and a **Logistic Regression** is trained on top. This is the same idea behind well-known click-through-rate models (e.g., GBDT + LR).

Adapted from and extending the scikit-learn ensemble documentation example.

## Approach

- **Data:** 80,000-sample synthetic binary classification set (`make_classification`), split into:
  - a set to train the tree ensembles
  - a separate set to train the linear models (avoids overfitting from reusing the same data)
  - a held-out test set
- **Models compared:**

| Model | Description |
|---|---|
| Random Forest | Standalone ensemble classifier |
| Gradient Boosting (GBDT) | Standalone boosted ensemble |
| RT embedding → LR | Unsupervised `RandomTreesEmbedding` + Logistic Regression |
| RF embedding → LR | Random Forest leaf indices, one-hot encoded + Logistic Regression |
| GBDT embedding → LR | Gradient Boosting leaf indices, one-hot encoded + Logistic Regression |

- **Evaluation:** ROC curves for all five models, plus a zoomed view of the high-performance top-left corner.

## Takeaways

Tree-based embeddings let a simple linear model capture non-linear feature interactions that it could not learn from the raw features. The ROC comparison shows how each embedding → LR pipeline performs relative to the standalone ensemble it was built from.

## How to run

```bash
pip install -r ../requirements.txt
jupyter notebook Gradient_Boost_Feature_Transformations_Ensemble_Trees.ipynb
```

## Skills demonstrated

Ensemble methods · scikit-learn `Pipeline` and `FunctionTransformer` · one-hot encoding · ROC analysis
