"""Shared data preparation code for the insurance claims project.

All notebooks import prepare_data from this file,
so both team members use the same version of the code.
"""
import pandas as pd

RANDOM_STATE = 42
TARGET = "annual_claim_cost"

# Fields that are known only after the policy is signed (data leakage).
# Fill this list from the leakage analysis (section 2.2).
LEAKAGE_COLUMNS = []


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get a raw DataFrame (dataset.csv format) and return the model features.

    The function must be deterministic and must work on one row or on many rows.
    """
    df = df.copy()
    drop_cols = [c for c in LEAKAGE_COLUMNS + [TARGET] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # TODO: engineered features (section 4)

    return df
