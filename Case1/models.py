# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # call it with squared=False to get RMSE
from sklearn.linear_model import LinearRegression, Ridge, LassoLars # Lasso takes cyclic or random selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

import warnings

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
lasso        = LassoLars(fit_intercept=False, normalize=False) # alpha defined later
elastic_net  = ElasticNet(fit_intercept=False, normalize=False) # alpha and l1 ratio defined later
Adaboost_knn = AdaBoostClassifier(KNeighborsRegressor(p=2, weights='distance'))

# -------------------All the things to save----------------------
num_ridge_lasso_lambdas = 100
ridge_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
num_elasticnet_alphas = 5
elasticnet_alphas = np.logspace(-4, 0, num_elasticnet_alphas)
elasticnet_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)

highest_knn_k = 10
knn_ks = np.arange(1, highest_knn_k)
highest_knn_k -1

Results = {'OLS': 
               {'RMSE': np.zeros(K)},
           'Ridge':
               {'lambdas': ridge_lambdas, 
                'RMSE': np.zeros((K, num_ridge_lasso_lambdas))},
           'KNN': 
               {'ks': knn_ks,
                'RMSE': np.zeros((K, highest_knn_k-1))},
           'Weighted KNN':
               {'ks': knn_ks, 
                'RMSE':np.zeros((K, highest_knn_k-1))},
           'Lasso': 
               {'lambdas': lasso_lambdas, 
                'RMSE': np.zeros((K, num_ridge_lasso_lambdas))},
           'Elastic Net': 
               {'lambdas': elasticnet_lambdas,
                'alphas': elasticnet_alphas, 
                'RMSE': np.zeros((K, num_ridge_lasso_lambdas, num_elasticnet_alphas))},
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
    print(f"Fold: {i}")
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
    Results['OLS']["RMSE"][i] = mean_squared_error(y_validation, y_pred, squared=False)
    
    for k in knn_ks:
        # -----------KNN---------------
        knn.set_params(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_validation)
        Results['KNN']["RMSE"][i, k-1] = mean_squared_error(y_validation, y_pred, squared=False)
        
        # -----------Weighted KNN---------------
        knn_weighted.set_params(n_neighbors = k)
        knn_weighted.fit(X_train, y_train)
        y_pred = knn_weighted.predict(X_validation)
        Results['Weighted KNN']["RMSE"][i, k-1] = mean_squared_error(y_validation, y_pred, squared=False)

    
    for j in range(num_ridge_lasso_lambdas):
        #------ RIDGE-----------------------        
        ridge.set_params(alpha=ridge_lambdas[j])
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_validation)
        Results['Ridge']["RMSE"][i, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        #--------LASSO-----------------------
        lasso.set_params(alpha= lasso_lambdas[j])
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_validation)
        Results['Lasso']["RMSE"][i, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        #-----------ElasticNet----------------
        for j2 in range(num_elasticnet_alphas):
            elastic_net.set_params(alpha=elasticnet_lambdas[j], l1_ratio = elasticnet_alphas[j2])
            with warnings.catch_warnings(): # convergence warnings
                warnings.simplefilter("ignore")
                elastic_net.fit(X_train, y_train)
            y_pred = elastic_net.predict(X_validation)
            Results['Elastic Net']["RMSE"][i, j, j2] = mean_squared_error(y_validation, y_pred, squared=False)

# --------------------------- Save results for analysis --------------------------
# Todo: bootstrap for evaluation with perfect lambdas

# %%
import pickle
import datetime as dt
import time

# ... versioning, huh?
time_now = dt.datetime.now()
results_file = open(f"results/results_rmse_{time.mktime(time_now.timetuple())}.pickle", "wb") 
pickle.dump(Results, results_file)







