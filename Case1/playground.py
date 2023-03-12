# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import *
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, LassoLars 

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# %%
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

# %%
# Better way of doing k fold cross validation
kf = KFold(n_splits=K)
k_neigh = 10
Error = np.zeros((K, k_neigh))

for i, (train_index, validation_index) in enumerate(kf.split(df)):
    # X_train = Xa[train_index]
    # y_train = y[train_index]
    # X_test = Xa[validation_index]
    # y_test = y[validation_index]

    df_train = df.iloc[train_index]
    df_validation = df.loc[validation_index]

    # prep
    standardizer_X = fit_standardizer_X(df_train[CONTINUOUS].values, df_new[CONTINUOUS].values)
    encoder        = fit_encoder(df[CATEGORICAL].values, df_new[CATEGORICAL].values)

    X_train = preprocess_X(df_train, standardizer_X, encoder)
    X_test  = preprocess_X(df_validation, standardizer_X, encoder)
    
    standardizer_y = fit_standardizer_y(df_train['y'].values)
    
    y_train = preprocess_y(df_train, standardizer_y)
    y_test = df_validation['y'].values

    for k in range(1,k_neigh+1):
        # Use Scikit KNN classifier, as you have already tried implementing it youself
        neigh = KNeighborsRegressor(n_neighbors=k, weights = 'uniform', metric = 'euclidean')
        neigh.fit(X_train, y_train)
        yhat = neigh.predict(X_test)
            
        # This time i use the MAE
        Error[i-1, k-1] = sum(np.abs(y_test - yhat)) / len(yhat)

# %%
    
E = np.mean(Error, axis = 0)

fig  = plt.figure(figsize=(10,10))
plt.scatter(list(range(1,k_neigh+1)), E, marker = '*')
# plt.axis([0, 11, 0.2, 0.6])
fig.suptitle("CV test error", fontsize=20)
plt.xlabel("K")
plt.ylabel("Error")
plt.show()

# %%
# pipline
CATEGORICAL = [c for c in df.columns if c.startswith("C")]
CONTINUOUS  = [x for x in df.columns if x.startswith("x")]

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

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("KNN_regressor", Ridge(fit_intercept=False, alpha = 10**(-1)))
    ]
)


# %%


X = df.iloc[:, df.columns != "y"]
y = df.y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%
y_pred = clf.predict(X_test)

# %%
# TREES
# Try to experiment with max_samples, max_features, number of modles, and other models
n_estimators = range(5,10)
max_depth = range(1,5)

#We do an outer loop over max_depth here ourselves because we cannot include in the CV paramgrid.
#Notice this is not a "proper" way to select the best max_depth but for the purpose of vizuallizing behaviour it should do
test_acc = np.zeros((len(n_estimators), len(max_depth)))
for i in max_depth:
    # Create and fit an AdaBoosted decision tree
    boost = AdaBoostClassifier(DecisionTreeRegressor(max_depth=i),
                             learning_rate=1.0)
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

# %%
fig, ax = plt.subplots(figsize=(15,15))

ax.plot(n_estimators, test_acc)
ax.set_xlabel('Maximum tree depth')
ax.set_ylabel('Mean test accuracy')
ax.legend(['MaxDepth=1','MaxDepth=2','MaxDepth=3','MaxDepth=4','MaxDepth=5',
           'MaxDepth=6','MaxDepth=7','MaxDepth=8','MaxDepth=9','MaxDepth=10'])

# %%
# Try to experiment with criterion, number of estimators, max_depth, min_samples_leaf
clf = RandomForestRegressor(bootstrap=True, oob_score=True,
                            criterion = 'gini',random_state=0)

# number of trees
n_estimators = range(5,101)
max_depth = range(1,11)
max_features = range(10,250,20)

# Try to add more of the parameters from the model and then add them to this dict to see how it affects the model.
param_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'max_features': max_features
}

rf_grid = GridSearchCV(estimator = clf, param_grid = param_grid,
                       cv = 5, verbose=2, n_jobs=-1)

# Fit the grid search model
rf_grid.fit(X, y)

#Save the results in a dataframe to disk
df_results = pd.DataFrame(rf_grid.cv_results_)
df_results.to_csv('CrossValidationResultsRandomForest.csv')

print(rf_grid.best_estimator_)

# %%
from h2o.tree import H2OTree
from h2o import H2OFrame
from h2o.estimators import H2OGradientBoostingEstimator

CATEGORICAL = [c for c in df.columns if c.startswith("C")]
CONTINUOUS  = [x for x in df.columns if x.startswith("x")]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_std=True, with_mean=True))
    ]
)
# creating new class
categorical_transformer_trees = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
    ]
)
preprocessor_trees = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, CONTINUOUS),
        ("cat", categorical_transformer_trees, CATEGORICAL),
    ]
)

clf_trees = Pipeline(steps=[("preprocessor", preprocessor_trees)])

X = df.iloc[:, df.columns != "y"]
y = df.y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = clf_trees.fit_transform(X_train)
X_test = clf_trees.transform(X_test) # note lack of fit

df_trees = df.iloc[:, df.columns != "y"]
df_trees[column for column in df_trees.columns if column != "y"] = X_train
df_trees['y'] = y_train

gbm = H2OGradientBoostingEstimator(ntrees=1)
gbm.train(x=[column for column in df_trees.columns if column != "y"],
          y='y', training_frame = H2OFrame(df_trees))
# Obtaining a tree is a matter of a single call
tree = H2OTree(model = gbm, tree_number = 0 , tree_class = "NO")
# tree.model_id
# tree.tree_number
# tree.tree_class
