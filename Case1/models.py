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
from sklearn.linear_model import ElasticNet

import warnings

from preprocessing import *

from h2o import H2OFrame
from h2o.estimators import H2OGradientBoostingEstimator

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
elastic_net_alphas = np.logspace(-4, 0, num_elasticnet_alphas)
highest_knn_k = 10


RMSE = {'OLS'  : np.zeros(K),
        'Ridge': np.zeros((K, num_ridge_lasso_lambdas)),
        'KNN': np.zeros(K),
        'Weighted KNN': np.zeros(K),
        'Lasso': np.zeros((K, num_ridge_lasso_lambdas)),
        'Elastic Net': np.zeros((K, num_ridge_lasso_lambdas, num_elasticnet_alphas)), 
       } # This could be a class

# ------------------Pipeline preprocess ---------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_std=True, with_mean=True))
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
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
        ("num", numeric_transformer, CONTINUOUS),
        ("cat", categorical_transformer_trees, CATEGORICAL),
    ]
)
clf = Pipeline(steps=[("preprocessor", preprocessor)])
clf_trees = Pipeline(steps=[("preprocessor", preprocessor_trees)])

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

    X_train_trees = clf_trees.fit_transform(X_train)
    X_validation_trees = clf.transform(X_validation)

    X_train = clf.fit_transform(X_train)
    X_validation = clf.transform(X_validation) # note lack of fit

    # ----------OLS--------------
    ols.fit(X_train, y_train)
    y_pred = ols.predict(X_validation)
    RMSE['OLS'][i] = mean_squared_error(y_validation, y_pred, squared=False)
    
    for k in range(1, highest_knn_k):
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

    
    for j in range(num_ridge_lasso_lambdas):
        #------ RIDGE-----------------------        
        ridge.set_params(alpha=ridge_lambdas[j])
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_validation)
        RMSE['Ridge'][i, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        #--------LASSO-----------------------
        lasso.set_params(alpha= lasso_lambdas[j])
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_validation)
        RMSE['Lasso'][i, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        #-----------ElasticNet----------------
        for j2 in range(num_elasticnet_alphas):
            elastic_net.set_params(alpha=ridge_lambdas[j], l1_ratio = elastic_net_alphas[j2])
            with warnings.catch_warnings(): # convergence warnings
                warnings.simplefilter("ignore")
                elastic_net.fit(X_train, y_train)
            y_pred = elastic_net.predict(X_validation)
            RMSE['Elastic Net'][i, j, j2] = mean_squared_error(y_validation, y_pred, squared=False)

    #-----------Tree based methods ----------------
    #-----------GradientBoostingTree----------------

    # H2O definition stuff
    df_h2o = H2OFrame(pd.DataFrame(X_train_trees, columns=X_train_trees.columns))
    df_h2o['y'] = y_train
    df_h2o_validation = H2OFrame(pd.DataFrame(X_validation_trees,
                                              columns=X_validation_trees.columns))
    df_h2o_validation['y'] = y_test

    for max_depth in ___:
        for number_trees in ____:
            gbm = H2OGradientBoostingEstimator(ntrees = number_trees, max_depth = max_depth)
            gbm.train(x = [column for column in df_h2o.columns if column != "y"], y = "y",
               training_frame = df_h2o)
            y_pred = gbm.predict(df_h2o_validation)
            var_importance = gbm.varimp(use_pandas=True).to_numpy()

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







