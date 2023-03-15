# %%
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error # call it with squared=False to get RMSE
from sklearn.linear_model import LinearRegression, Ridge, LassoLars # Lasso takes cyclic or random selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

import warnings
from preprocessing import *

from h2o import H2OFrame
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator
import h2o


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

# -------------------Define models -------------------------------
ols          = LinearRegression(fit_intercept=False) # no intercept, as we center the data
knn          = KNeighborsRegressor(p=2, weights='uniform') # p=2 uses eucledian distance
knn_weighted = KNeighborsRegressor(p=2, weights='distance') # p=2 uses eucledian distance
ridge        = Ridge(fit_intercept=False) # alpha defined later
lasso        = LassoLars(fit_intercept=False, normalize=False) # alpha defined later
elastic_net  = ElasticNet(fit_intercept=False) # alpha and l1 ratio defined later
gbm          = H2OGradientBoostingEstimator()
Adaboost_knn = AdaBoostRegressor(KNeighborsRegressor(p=2, weights='distance'))
Adaboost_trees = AdaBoostRegressor(DecisionTreeRegressor())
rf           = H2ORandomForestEstimator(balance_classes=True, seed=1234) #note, it's distributed

# -------------------All the things to save----------------------

if not test:
    num_ridge_lasso_lambdas = 100
    ridge_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
    lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
    num_elasticnet_alphas = 5
    elasticnet_alphas = np.logspace(-4, 0, num_elasticnet_alphas)
    elasticnet_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)

    highest_knn_k = 10
    knn_ks = np.arange(1, highest_knn_k)
    highest_knn_k -1

    max_depth_GradientBoost = np.arange(2, 6, 1)
    number_trees_GradientBoost = np.arange(5, 50, 3)
    learning_rate_GradientBoost = np.array([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3])

    noise_features_number = 5
    noise_variance = 0.5

    ntrees = 200
    max_depth = 30
    max_depth_rf = np.arange(10, max_depth, 5)
    number_trees_rf = np.arange(10, ntrees, 10)

    # #### AdaBoost 
    n_estimators = np.arange(10, 50, 5)
    learning_rate_adaboost = np.arange(0.01, 0.3, 0.01)
    knn_ks_adaboost = np.arange(10, 30, 3)
    max_depth_adaboost = np.arange(2,11)
else:
    num_ridge_lasso_lambdas = 2
    ridge_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
    lasso_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)
    num_elasticnet_alphas = 2
    elasticnet_alphas = np.logspace(-4, 0, num_elasticnet_alphas)
    elasticnet_lambdas = np.logspace(-4, 4, num_ridge_lasso_lambdas)

    highest_knn_k = 2
    knn_ks = np.arange(1, highest_knn_k)
    highest_knn_k -1

    max_depth_GradientBoost = range(5, 6, 1)
    number_trees_GradientBoost = range(3, 4, 1)
    learning_rate_GradientBoost = np.arange(0.1, 0.2, 0.1)
    noise_features_number = 5
    noise_variance = 0.5

    ntrees = 10
    max_depth = 3
    max_depth_rf = range(9, ntrees, 1)
    number_trees_rf = range(2, max_depth, 1)

    #### AdaBoost 
    n_estimators = range(3, 4, 1)
    learning_rate_adaboost = np.arange(0.1, 0.3, 0.1)
    knn_ks_adaboost = range(5, 6, 1)
    max_depth_adaboost = range(2,3)


Results = {
    'OLS': 
        {'RMSE': np.zeros(K)},
    'Ridge':
        {'lambdas': ridge_lambdas, 
        'RMSE': np.zeros((K, num_ridge_lasso_lambdas))},
    'KNN': 
        {'ks': knn_ks,
        'RMSE': np.zeros((K, highest_knn_k-1))},
    'Weighted_KNN':
        {'ks': knn_ks, 
        'RMSE':np.zeros((K, highest_knn_k-1))},
    'Lasso': 
        {'lambdas': lasso_lambdas, 
        'RMSE': np.zeros((K, num_ridge_lasso_lambdas))},
    'Elastic_Net': 
        {'lambdas': elasticnet_lambdas,
        'alphas': elasticnet_alphas, 
        'RMSE': np.zeros((K, num_ridge_lasso_lambdas, num_elasticnet_alphas))},
    'Gradient_Boosting_Tree': 
        {'max_depth': max_depth_GradientBoost,
        'number_trees': number_trees_GradientBoost, 
        'learning_rate': learning_rate_GradientBoost,
        'RMSE': np.zeros((K, len(max_depth_GradientBoost), len(number_trees_GradientBoost),
                            len(learning_rate_GradientBoost)))},
    'Random_Forest': 
        {'max_depth': max_depth_rf,
        'number_trees': number_trees_rf, 
        'RMSE': np.zeros((K, len(max_depth_rf), len(number_trees_rf)))},
    'Adaboost_knn': 
        {'ks': knn_ks_adaboost,
        'n_estimators': n_estimators, 
        'learning_rate': learning_rate_adaboost,
        'RMSE': np.zeros((K, len(knn_ks_adaboost), len(n_estimators),
                            len(learning_rate_adaboost)))},
    'Adaboost_trees': 
       {'max_depth': max_depth_adaboost,
        'number_trees': n_estimators, 
        'learning_rate': learning_rate_adaboost,
        'RMSE': np.zeros((K, len(max_depth_adaboost), len(n_estimators),
                          len(learning_rate_adaboost)))},   
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

# -------------------- Train and find params---------------------------
kf = KFold(n_splits=K)
h2o.init()

for fold_index, (train_index, validation_index) in enumerate(kf.split(df)):
    print(f"Fold: {fold_index}")
    # split
    df_train = df.iloc[train_index]
    df_validation = df.loc[validation_index]
    
    # prep
    X_train_initial = df_train.iloc[:, df_train.columns != "y"]
    X_validation = df_validation.iloc[:, df_validation.columns != "y"]

    y_train = df_train.y.values
    y_validation = df_validation.y.values

    X_train_trees = clf_trees.fit_transform(X_train_initial)
    X_validation_trees = clf_trees.transform(X_validation) # note lack of fit

    X_train = clf.fit_transform(X_train_initial)
    X_validation = clf.transform(X_validation) # note lack of fit

    # ----------OLS--------------
    ols.fit(X_train, y_train)
    y_pred = ols.predict(X_validation)
    Results['OLS']["RMSE"][fold_index] = mean_squared_error(y_validation, y_pred, squared=False)
    
    for k in knn_ks:
        # -----------KNN---------------
        knn.set_params(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_validation)
        Results['KNN']["RMSE"][fold_index, k-1] = mean_squared_error(y_validation, y_pred, squared=False)
        
        # -----------Weighted KNN---------------
        knn_weighted.set_params(n_neighbors = k)
        knn_weighted.fit(X_train, y_train)
        y_pred = knn_weighted.predict(X_validation)
        Results['Weighted_KNN']["RMSE"][fold_index, k-1] = mean_squared_error(y_validation, y_pred, squared=False)

    
    for j in range(num_ridge_lasso_lambdas):
        #------ RIDGE-----------------------        
        ridge.set_params(alpha=ridge_lambdas[j])
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_validation)
        Results['Ridge']["RMSE"][fold_index, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        #--------LASSO-----------------------
        lasso.set_params(alpha= lasso_lambdas[j])
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_validation)
        Results['Lasso']["RMSE"][fold_index, j] = mean_squared_error(y_validation, y_pred, squared=False)
        
        #-----------ElasticNet----------------
        for j2 in range(num_elasticnet_alphas):
            elastic_net.set_params(alpha=elasticnet_lambdas[j], l1_ratio = elasticnet_alphas[j2])
            with warnings.catch_warnings(): # convergence warnings
                warnings.simplefilter("ignore")
                elastic_net.fit(X_train, y_train)
            y_pred = elastic_net.predict(X_validation)
            Results['Elastic_Net']["RMSE"][fold_index, j, j2] = mean_squared_error(y_validation, y_pred, squared=False)

    # -----------Tree based methods ----------------
    # H2O definition stuff

    df_h2o = pd.DataFrame(X_train_trees, columns=X_train_initial.columns)
    df_h2o['y'] = y_train
    df_h2o_validation = pd.DataFrame(X_validation_trees,
                                              columns=X_train_initial.columns)
    df_h2o_validation['y'] = y_validation

    # noise_names = ["noise_" + str(i) for i in range(noise_features_number)]

    # noise = noise_variance*np.random.randn(noise_features_number, df_h2o.shape[0]).T
    # df_h2o[noise_names] = noise

    # noise_validation = noise_variance*np.random.randn(noise_features_number, df_h2o.shape[0]).T
    # df_h2o_validation[noise_names] = noise_validation

    df_h2o = H2OFrame(df_h2o)
    df_h2o_validation = H2OFrame(df_h2o_validation)
    columns_h2o = [column for column in df_h2o.columns if column != "y"]

    #-----------GradientBoostingTree----------------
    for max_depth_index in range(len(Results["Gradient_Boosting_Tree"]["max_depth"])):
        for number_trees_index in range(len(Results["Gradient_Boosting_Tree"]["number_trees"])):
            for learning_rate_index in range(len(Results["Gradient_Boosting_Tree"]["learning_rate"])):

                max_depth = Results["Gradient_Boosting_Tree"]["max_depth"][max_depth_index]
                number_trees = Results["Gradient_Boosting_Tree"]["number_trees"][number_trees_index]
                learning_rate = Results["Gradient_Boosting_Tree"]["learning_rate"][learning_rate_index]

                print(max_depth, number_trees, learning_rate)
                gbm.set_params(ntrees = number_trees, max_depth = max_depth,
                                learn_rate = learning_rate)
                gbm.train(x = columns_h2o, y = "y",
                    training_frame = df_h2o)
            
                # save column name, relative imporatnce and percentage imporatnce
                # var_importance = gbm.varimp(use_pandas=True).iloc[:, [0, 1, 3]].to_numpy()
                # rmse_list = gbm.score_history().loc[:, ["training_rmse", "validation_rmse"]].to_numpy()

                y_pred = gbm.predict(df_h2o_validation).as_data_frame().to_numpy().squeeze()
                Results['Gradient_Boosting_Tree']["RMSE"][fold_index, max_depth_index, number_trees_index, learning_rate_index] = \
                    mean_squared_error(y_validation, y_pred, squared=False)
                # Results['Gradient_Boosting_Tree']["Var_importance"][max_depth_index, number_trees_index, learning_rate_index,:,:] = \
                #     var_importance

    #-----------RandomForest----------------
    for max_depth_index in range(len(Results["Random_Forest"]["max_depth"])):
        for number_trees_index in range(len(Results["Random_Forest"]["number_trees"])):

            max_depth = Results["Random_Forest"]["max_depth"][max_depth_index]
            number_trees = Results["Random_Forest"]["number_trees"][number_trees_index]

            rf.set_params(ntrees = number_trees, max_depth = max_depth)
            rf.train(x = columns_h2o, y = "y",
                training_frame = df_h2o, validation_frame = df_h2o_validation)
            
            y_pred = rf.predict(df_h2o_validation).as_data_frame().to_numpy().squeeze()
            rmse = mean_squared_error(y_validation, y_pred, squared=False)
            Results['Random_Forest']["RMSE"][fold_index, max_depth_index, number_trees_index] = rmse
         
    #-----------AdaBoost_KNN---------------- 
    for ks_index in range(len(Results["Adaboost_knn"]["ks"])):
        for n_estimators_index in range(len(Results["Adaboost_knn"]["n_estimators"])):
            for learning_rate_index in range(len(Results["Adaboost_knn"]["learning_rate"])):

                ks = Results["Adaboost_knn"]["ks"][ks_index]
                n_estimators = Results["Adaboost_knn"]["n_estimators"][n_estimators_index]
                learning_rate = Results["Adaboost_knn"]["learning_rate"][learning_rate_index]

                Adaboost_knn.set_params(learning_rate=learning_rate, n_estimators = n_estimators)
                Adaboost_knn.estimator.set_params(n_neighbors = ks)
                Adaboost_knn.fit(X_train, y_train)
                y_pred = Adaboost_knn.predict(X_validation)
                Results['Adaboost_knn']["RMSE"][fold_index, ks_index, n_estimators_index, learning_rate_index] =\
                    mean_squared_error(y_validation, y_pred, squared=False)

    #-----------AdaBoost_Trees---------------- 
    print("Adaboost_trees")
    for max_depth_index in range(len(Results["Adaboost_trees"]["max_depth"])):
        for number_trees_index in range(len(Results["Adaboost_trees"]["number_trees"])):
            for learning_rate_index in range(len(Results["Adaboost_trees"]["learning_rate"])):

                max_depth = Results["Adaboost_trees"]["max_depth"][max_depth_index]
                number_trees = Results["Adaboost_trees"]["number_trees"][number_trees_index]
                learning_rate = Results["Adaboost_trees"]["learning_rate"][learning_rate_index]

                Adaboost_trees.set_params(learning_rate=learning_rate, n_estimators = n_estimators)
                Adaboost_trees.estimator.set_params(max_depth = max_depth)

                Adaboost_trees.fit(X_train, y_train)
                y_pred = Adaboost_trees.predict(X_validation)
                Results['Adaboost_trees']["RMSE"][fold_index, max_depth_index, number_trees_index, learning_rate_index] =\
                    mean_squared_error(y_validation, y_pred, squared=False)
                             

                   
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







