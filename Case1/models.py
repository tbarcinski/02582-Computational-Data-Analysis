
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # call it with squared=False to get RMSE
from sklearn.linear_model import LinearRegression, Ridge, LassoLars # Lasso takes cyclic or random selection
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.regression.linear_model import OLS

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


# -------------------All the things to save----------------------
num_ridge_lasso_lambdas = 100
ridge_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)


RMSE = {'OLS'  : np.zeros(K),
        'Ridge': np.zeros((K, num_ridge_lasso_lambdas)),
        'KNN': np.zeros(K),# Anna
        'Weighted KNN': np.zeros(K), # Anna
        'Lasso': np.zeros((K, num_ridge_lasso_lambdas)), # Tymek,
        'Elastic Net': None, # Tymek
       } # This could be a class


# -------------------- Train and find params---------------------------

fold_idxs = np.arange(0, N)%K
fold_idxs = np.random.permutation(fold_idxs) #shuffle

for i in range(K):
    # split
    df_train = df.loc[fold_idxs !=i]
    df_test = df.loc[fold_idxs == i]
    
    # prep
    standardizer_X = fit_standardizer_X(df_train[CONTINUOUS].values, df_new[CONTINUOUS].values)
    encoder        = fit_encoder(df[CATEGORICAL].values, df_new[CATEGORICAL].values)
    
    X_train = preprocess_X(df_train, standardizer_X, encoder)
    X_test  = preprocess_X(df_test, standardizer_X, encoder)
    
    standardizer_y = fit_standardizer_y(df_train['y'].values)
    
    y_train = preprocess_y(df_train, standardizer_y)
    y_test = df_test['y'].values
    
    # ----------OLS--------------
    ols.fit(X_train, y_train)
    y_pred = ols.predict(X_test)
    rmse = mean_squared_error(y_test, postprocess_y(y_pred, standardizer_y), squared=False)
    RMSE['OLS'][i] = rmse
    
    # -----------KNN---------------
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    RMSE['KNN'][i] = mean_squared_error(y_test,postprocess_y(y_pred, standardizer_y), squared=False)
    
    # -----------Weighted KNN---------------
    knn_weighted.fit(X_train, y_train)
    y_pred = knn_weighted.predict(X_test)
    RMSE['Weighted KNN'][i] = mean_squared_error(y_test, postprocess_y(y_pred, standardizer_y), squared=False)
    
    #------ RIDGE-----------------------
    for j in range(num_ridge_lasso_lambdas):
        
        ridge.set_params(alpha=ridge_lambdas[j])
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        RMSE['Ridge'][i, j] = mean_squared_error(y_test, postprocess_y(y_pred, standardizer_y), squared=False)
        
        lasso.set_params(alpha= lasso_lambdas[j])
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        RMSE['Lasso'][i, j] = mean_squared_error(y_test, postprocess_y(y_pred, standardizer_y), squared=False)
        
# --------------------------- Save results for analysis --------------------------
# Todo: bootstrap for evaluation with perfect lambdas

import pickle
import datetime as dt

# ... versioning, huh?
results = open(f"results/results_rmse_{dt.datetime.now()}.pickle", "wb") 
pickle.dump(RMSE, results)







