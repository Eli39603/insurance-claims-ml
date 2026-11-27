# Predicting Car Insurance Claim Costs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eli39603/insurance-claims-ml/blob/main/notebook.ipynb)

Machine Learning final project.

The model predicts `annual_claim_cost`: the total cost of claims (in ILS) that a driver will cause the insurance company in the next policy year. The company can use this prediction to decide whether to insure a driver.

The model uses **only fields that are known before the policy is signed**. Fields that are known only after a claim (data leakage) are removed.

## Model

- **Input:** one row per driver per policy year (raw `dataset.csv` format).
- **Target:** `annual_claim_cost` (0 if there was no claim).
- **Preprocessing:** `prepare_data(df)` cleans the data and builds the engineered features. Imputation, scaling and encoding are inside a scikit-learn `Pipeline`, so they are fit only on the training fold in cross validation.
- **Models:**
  - Elastic Net (linear), with `alpha` and `l1_ratio` tuned in cross validation. This model is saved as `model.pkl`.
  - Random Forest.
- **Evaluation:** 10-fold cross validation. Metrics: RMSE, MAE, R² (mean and standard deviation over the folds).
- **Analysis:** data exploration, age strategy, complexity curve, error analysis and fairness analysis. See `notebook.ipynb` and `report.pdf`.

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Elastic Net | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO |

## Repository files

| File | Content |
|---|---|
| `notebook.ipynb` | Full code: `prepare_data`, training, evaluation, error analysis and fairness analysis |
| `model.pkl` | Trained Elastic Net model (joblib) |
| `report.pdf` | Project report |
| `requirements.txt` | Python libraries and versions |

## How to run

1. Install Python 3.10 or later.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Put `dataset.csv` in the project folder.
4. Open `notebook.ipynb` (in Jupyter or with the Colab badge above).
5. Run all cells from top to bottom. The code uses `random_state=42` for reproducible results.

## Predict on new data

```python
import pandas as pd
import joblib

model = joblib.load("model.pkl")
df_test = pd.read_csv("test.csv")   # same structure as dataset.csv, without the target column
X = prepare_data(df_test)           # defined in notebook.ipynb
y_pred = model.predict(X)
```

## Authors

- Eliran Levi
- Mali Bigel
