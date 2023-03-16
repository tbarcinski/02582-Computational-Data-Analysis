
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error # call it with squared=False to get RMSE
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from preprocessing import *

test = False

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

if not test:

    num_elasticnet_alphas = 10
    num_ridge_lasso_lambdas = 100

    elasticnet_alphas = np.logspace(-4, 0, num_elasticnet_alphas, endpoint=False)
    elasticnet_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
    
else:
    num_elasticnet_alphas = 3
    num_ridge_lasso_lambdas = 5

    elasticnet_alphas = np.logspace(-4, 0, num_elasticnet_alphas, endpoint=False)
    elasticnet_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)


Results = {
    'Elastic_Net': 
        {'lambdas': elasticnet_lambdas,
        'alphas': elasticnet_alphas, 
        'Result': np.zeros((5, 3))}, # lambda, alpha, rmse (for each fold)
           }

# ---------------------- Model --------------------------------

elastic_net = ElasticNetCV(l1_ratio = elasticnet_alphas, alphas = elasticnet_lambdas, fit_intercept=False, random_state=6)



kf = KFold(n_splits=K, random_state=42, shuffle=True)

for fold_index, (train_index, validation_index) in enumerate(kf.split(df)):
    print(f"Fold: {fold_index}")
    # split
    df_train = df.iloc[train_index]
    df_validation = df.loc[validation_index]
    
    # prep
    X_train_initial = df_train.iloc[:, df_train.columns != "y"]
    X_validation = df_validation.iloc[:, df_validation.columns != "y"]
    X_new = df_new

    y_train = df_train.y.values
    y_validation = df_validation.y.values
    
    clf.fit(pd.concat((X_train_initial, X_new), axis=0))
    X_train = clf.transform(X_train_initial)
    X_validation = clf.transform(X_validation)
    
    with warnings.catch_warnings(): # convergence warnings
        warnings.simplefilter("ignore")
        elastic_net.fit(X_train, y_train)
    y_pred = elastic_net.predict(X_validation)
    rmse = mean_squared_error(y_validation, y_pred, squared=False)
    Results['Elastic_Net']["Result"][fold_index] = np.array([elastic_net.alpha_, elastic_net.l1_ratio_, rmse])
    
    
    
# TODO: 
# 1. add epe calc - save EPE
# 2. add bootstrap + get epe - save EPE
# 3. find lambda* and alpha* by gridsearch (non-nested cv)
# 4. retrain on all data and save predictions

    
    

    
    
    
