# %%
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("case1Data.txt", sep=', ')
df.columns = [c.replace(' ', '') for c in df.columns]
CATEGORICAL = [c for c in df.columns if c.startswith("C")]
CONTINUOUS  = [x for x in df.columns if x.startswith("x")]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler(with_std=True, with_mean=True)),
        ("feature_extration", PolynomialFeatures(degree=1, with_bias=True))
    ]
)

numeric_transformer_trees = Pipeline(
    steps=[
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler(with_std=True, with_mean=True)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
categorical_transformer_trees = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, CONTINUOUS),
        ("cat", categorical_transformer, CATEGORICAL),
    ]
)
preprocessor_trees = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_trees, CONTINUOUS),
        ("cat", categorical_transformer_trees, CATEGORICAL),
    ]
)


clf = Pipeline(steps=[("preprocessor", preprocessor)])
clf_trees = Pipeline(steps=[("preprocessor", preprocessor_trees)])

