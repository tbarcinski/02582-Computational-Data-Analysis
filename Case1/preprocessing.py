
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

df = pd.read_csv("case1Data.txt", sep=', ')
df.columns = [c.replace(' ', '') for c in df.columns]
CATEGORICAL = [c for c in df.columns if c.startswith("C")]
CONTINUOUS  = [x for x in df.columns if x.startswith("x")]

def replace_missing_continouous(X: np.array):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X
    
def replace_missing_categorical(C: np.array) -> np.array:
    imputer = SimpleImputer(strategy='most_frequent')
    C = imputer.fit_transform(C)
    return C

def fit_encoder(C, C_new, categories='auto'):
    #TOASK: should we use all overall categories or only the one in a feature. 
    # 'auto' infers categories ie. c_2 would only have two categories
    
    combined = np.concatenate((C, C_new), axis=0)
    combined = replace_missing_categorical(combined)
    
    # sparse=False to avoid sparse matrix because todense becomes "matrix" wich is deprecated with some sklearn models
    enc = OneHotEncoder(categories=categories, sparse=False) 
    enc.fit(C)
    return enc

def fit_standardizer_X(X, X_new) -> StandardScaler:
    # we should fit the standardizer before replacing missing
    # so the bias from that doesn't affect normalization
    
    combined = np.concatenate((X, X_new), axis=0)
    
    standardizer = StandardScaler(with_std=True, with_mean=True)
    standardizer.fit(combined)
    
    return standardizer

def fit_standardizer_y(y):
    standardizer = StandardScaler(with_std=False, with_mean=True)
    standardizer.fit(y[:, np.newaxis])
    return standardizer

def preprocess_y(df, standardizer):
    y_standard = standardizer.transform(df[['y']].values)
    return y_standard.flatten()

def preprocess_X(df, standardizer, encoder):
    X = df[CONTINUOUS].values
    C = df[CATEGORICAL].values
    
    X_nonan = replace_missing_continouous(X)
    X_standard = standardizer.transform(X_nonan)

    C_nonan = replace_missing_categorical(C)
    C_onehot = encoder.transform(C_nonan) # TOASK: any advantage in standardizing this?

    combined = np.concatenate((X_standard, C_onehot), axis=1)

    return combined

def postprocess_y(y, standardizer):
    y_orig = standardizer.inverse_transform([y])
    return y_orig[0]




if __name__ == "__main__":
    print("preprocessing example")


    # If it's understood as categorical pd.get_dummies work sensibily TOASK: should we use all overall categories or only the one in a feature
    df[CATEGORICAL].astype(pd.CategoricalDtype(categories=set(df[CATEGORICAL].stack())))

    df_new = pd.read_csv("case1Data_Xnew.txt", sep=', ') # for competition, without y
    df_new.columns = [c.replace(' ', '') for c in df_new.columns]

    df_new[CATEGORICAL].astype(pd.CategoricalDtype(categories=set(df_new[CATEGORICAL].stack())))
    
    # example use - in practice we must preprocess within the fold!
    standardizer = fit_standardizer_X(df[CONTINUOUS].values, df_new[CONTINUOUS].values)
    encoder = fit_encoder(df[CATEGORICAL].values, df_new[CATEGORICAL].values)
    X_train = preprocess_X(df, standardizer, encoder)

    standardizer = fit_standardizer_y(df['y'].values)
    y_train = preprocess_y(df, standardizer)
    print('done')
