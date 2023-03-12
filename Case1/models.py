# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # call it with squared=False to get RMSE
from sklearn.linear_model import LinearRegression, Ridge, LassoLars # Lasso takes cyclic or random selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor

from preprocessing import *

# ----------------- Read data-------------------------------

df = pd.read_csv("case1Data.txt", sep=', ', engine='python')
df.columns = [c.replace(' ', '') for c in df.columns]


CATEGORICAL = [c for c in df.columns if c.startswith("C")]
CONTINUOUS  = [x for x in df.columns if x.startswith("x")]

# If it's understood as categorical pd.get_dummies work sensibily TOASK: should we use all overall categories or only the one in a feature
df[CATEGORICAL].astype(pd.CategoricalDtype(categories=set(df[CATEGORICAL].stack())))

df_new = pd.read_csv("case1Data_Xnew.txt", sep=', ', engine='python') # for competition, without y
df_new.columns = [c.replace(' ', '') for c in df_new.columns]

df_new[CATEGORICAL].astype(pd.CategoricalDtype(categories=set(df_new[CATEGORICAL].stack())))

df.head()

# ------------------Constants ---------------------------------

N = len(df) # Total number of observations in "case1Data.txt"
K = 5 # For K-fold cross validation

# -------------------Define models -------------------------------
ols          = LinearRegression(fit_intercept=False) # no intercept, as we center the data
knn          = KNeighborsRegressor(p=2, weights='uniform') # p=2 uses eucledian distance
knn_weighted = KNeighborsRegressor(p=2, weights='distance') # p=2 uses eucledian distance
ridge        = Ridge(fit_intercept=False) # alpha defined later
lasso        = LassoLars( fit_intercept=False, normalize=False) # alpha defined later
Adaboost_tree = 
Adaboost_knn = AdaBoostClassifier(KNeighborsRegressor(p=2, weights='distance'))
RandomForest = 

# -------------------All the things to save----------------------
num_ridge_lasso_lambdas = 100
ridge_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
highest_knn_k = 10

, learning_rate=1.0


RMSE = {'OLS'  : np.zeros(K),
        'Ridge': np.zeros((K, num_ridge_lasso_lambdas)),
        'KNN': np.zeros(K),# Anna
        'Weighted KNN': np.zeros(K), # Anna
        'Lasso': np.zeros((K, num_ridge_lasso_lambdas)), # Tymek,
        'Elastic Net': None, # Tymek
       } # This could be a class

# ------------------Pipeline preprocess ---------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_std=True, with_mean=True))
    ]
)
# creating new class
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, CONTINUOUS),
        ("cat", categorical_transformer, CATEGORICAL),
    ]
)

clf = Pipeline(steps=[("preprocessor", preprocessor)])

# %%
# -------------------- Train and find params---------------------------
kf = KFold(n_splits=K)

for i, (train_index, validation_index) in enumerate(kf.split(df)):
    # split
    df_train = df.iloc[train_index]
    df_validation = df.loc[validation_index]
    
    # prep
    X_train = df_train.iloc[:, df_train.columns != "y"]
    X_validation = df_validation.iloc[:, df_validation.columns != "y"]

    y_train = df_train.y.values
    y_validation = df_validation.y.values

    X_train = clf.fit_transform(X_train)
    X_validation = clf.transform(X_validation) # note lack of fit
    
    # ----------OLS--------------
    ols.fit(X_train, y_train)
    y_pred = ols.predict(X_validation)
    RMSE['OLS'][i] = mean_squared_error(y_validation, y_pred, squared=False)
    
    for k in range(highest_knn_k):
        # -----------KNN---------------
        knn.set_params(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_validation)
        RMSE['KNN'][i] = mean_squared_error(y_validation, y_pred, squared=False)
        
        # -----------Weighted KNN---------------
        knn_weighted.set_params(n_neighbors = k)
        knn_weighted.fit(X_train, y_train)
        y_pred = knn_weighted.predict(X_validation)
        RMSE['Weighted KNN'][i] = mean_squared_error(y_validation, y_pred, squared=False)

    #------ RIDGE-----------------------
    for j in range(num_ridge_lasso_lambdas):
        
        ridge.set_params(alpha=ridge_lambdas[j])
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_validation)
        RMSE['Ridge'][i, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        lasso.set_params(alpha= lasso_lambdas[j])
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_validation)
        RMSE['Lasso'][i, j] = mean_squared_error(y_validation, y_pred, squared=False)
    
    for i in max_depth:
        for j in learning
    # Create and fit an AdaBoosted decision tree

    pipeline_testing = Pipeline(
        steps=[
                ("preprocessor", preprocessor),
                ("regressor", boost)
            ]
        )

    param_grid = {
        'n_estimators': n_estimators
    }
    boost_grid = GridSearchCV(estimator = pipeline_testing, param_grid = param_grid,
                              cv = 5, verbose=2, n_jobs=-1)

    # Fit the grid search model
    boost_grid.fit(X, y)

    test_acc[:,i-1] = boost_grid.cv_results_['mean_test_score']
    
# --------------------------- Save results for analysis --------------------------
# Todo: bootstrap for evaluation with perfect lambdas

# %%
import pickle
import datetime as dt
import time

# ... versioning, huh?
time_now = dt.datetime.now()
results = open(f"results/results_rmse_{time.mktime(time_now.timetuple())}.pickle", "wb") 
pickle.dump(RMSE, results)







