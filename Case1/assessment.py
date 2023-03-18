import datetime as dt
import time
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error # call it with squared=False to get RMSE
from sklearn.linear_model import ElasticNetCV, LassoCV, LassoLars
from sklearn.model_selection import KFold
from sklearn.utils import resample
from preprocessing import *
time_now = dt.datetime.now()
test = True

def save_epe(Results):
    results_file = open(f"results/results_epe_{time.mktime(time_now.timetuple())}.pickle", "wb") 
    pickle.dump(Results, results_file)
    results_file.close()

def get_train_valid_split(df, train_index, validation_index):
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
    
    return X_train, X_validation, y_train, y_validation
    
    
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

    num_ridge_lasso_lambdas = 100
    lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
    
else:
    K = 2
    num_ridge_lasso_lambdas = 3
    lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)


Results = {
    'Lasso': {
        'lambdas': lasso_lambdas,
        'RMSE_CV': np.zeros((K, 2)),
        'RMSE_BS:' : np.zeros((K, 2)),
        'RMSE_tuning': np.zeros((K, num_ridge_lasso_lambdas)),
    }
}

# ---------------------- Cross validation  --------------------------------

model_epe = LassoCV(alphas=lasso_lambdas, fit_intercept=False, random_state=6)
model_final = LassoLars(fit_intercept=False)



kf = KFold(n_splits=K, random_state=42, shuffle=True)

for fold_index, (train_index, validation_index) in enumerate(kf.split(df)):
    print(f"Fold: {fold_index}")

    X_train, X_validation, y_train, y_validation = get_train_valid_split(df, train_index, validation_index)
    model_epe.fit(X_train, y_train)
    y_pred = model_epe.predict(X_validation)
    rmse = mean_squared_error(y_validation, y_pred, squared=False)
    Results['Lasso']["RMSE_CV"][fold_index] = np.array([model_epe.alpha_, rmse])
    
save_epe(Results)

# ------------------------------ final model ----------------------------------------
for fold_index, (train_index, validation_index) in enumerate(kf.split(df)):
    print(f"Fold: {fold_index}")
    X_train, X_validation, y_train, y_validation = get_train_valid_split(df, train_index, validation_index)
    
    for j, lambda_ in enumerate(lasso_lambdas):
        model_final.set_params(alpha= lasso_lambdas[j])
        model_final.fit(X_train, y_train)
        y_pred = model_final.predict(X_validation)
        rmse = mean_squared_error(y_validation, y_pred, squared=False)
        Results['Lasso']["RMSE_tuning"][fold_index, j] = np.array([rmse])

save_epe(Results)

# ----------------------------- retrain and get preds --------------------------------------

# get best (could also be one std error rule)

best_lambda_idx = Results['Lasso']["RMSE_tuning"].mean(axis=0).argmin()
best_lambda = lasso_lambdas[best_lambda_idx]

# prep
X = df.iloc[:, df.columns != "y"]
X_new = df_new

y = df.y.values

clf.fit(pd.concat((X, X_new), axis=0))
X = clf.transform(X)
X_new = clf.transform(X_new)

model_final.set_params(alpha= best_lambda)
model_final.fit(X, y)

y_pred = model_final.predict(X_new)

np.savetxt("predictions_s220817.txt",y_pred)

    


                                   

        

    

    
    
    
