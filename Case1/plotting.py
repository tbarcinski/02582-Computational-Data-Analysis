
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_best_three_params(Results, method):
    res = Results[method]
    K = res['RMSE'].shape[0]
    best_rmse_folds = np.zeros(K)
    best_params_folds = np.zeros((K, 3))
    first_param = "max_depth" if "tree" in method.lower() else "ks"
    second_param = "number_trees" if "tree" in method.lower() else "n_estimators"
    third_param = "learning_rate"
    for k in range(K):
        fold = res['RMSE'][k]
        # get best per max depth
        best_rmse = np.zeros((len(res[first_param]), 1))
        best_ntrees_lr = np.zeros((len(res[first_param]), 2))
        for i in range(len(res[first_param])):
            #print("max_depth:", res['max_depth'][i])
            A = fold[i, :, :]
            #print("norm:", np.linalg.norm(A), "min:", A.min())
            ri, ci = A.argmin()//A.shape[1], A.argmin()%A.shape[1]
            best_rmse[i] = A[ri, ci]
            best_ntrees_lr[i, :] = np.array([res[second_param][ri], res[third_param][ci]])
            #print("best ntrees lr:", np.array([res['number_trees'][ri], res['learning_rate'][ci]]))
        # get best across all max depths

        best_in_all_maxdepths_idx = best_rmse.argmin()
        best_in_all_maxdepths = best_rmse.min()
        best_rmse_folds[k] = best_in_all_maxdepths
        best_params_folds[k, 0] = res[first_param][best_in_all_maxdepths_idx]
        best_params_folds[k, 1:] = best_ntrees_lr[best_in_all_maxdepths_idx]
    return best_params_folds, best_rmse_folds

def get_best_one_param(Results, method):
    best_rmse = Results[method]['RMSE'].min(axis=1)
    best_idxs = Results[method]['RMSE'].argmin(axis=1)
    param = 'ks' if 'KNN' in method else 'lambdas'
    best_lambdas = np.array([Results[method][param][idx] for idx in best_idxs])
    return (best_lambdas, best_rmse)
    
    
def get_best_two_params(Results, method):
    res = Results[method]
    param1 = 'lambdas' if method=='Elastic_Net' else 'max_depth'
    param2 = 'alphas' if method=='Elastic_Net' else 'number_trees'
    K = res['RMSE'].shape[0]
    best_rmse = np.zeros(K)
    best_lambda_alpha = np.zeros((K, 2))
    for i in range(K):
        A = res["RMSE"][i]
        ri, ci = A.argmin()//A.shape[1], A.argmin()%A.shape[1]
        best_rmse[i] = A[ri, ci]
        best_lambda_alpha[i, :] = np.array([res[param1][ri], res[param2][ci]])
    return best_lambda_alpha, best_rmse

def get_best_rmse(Results, method):
    num_params = len(Results[method].keys()) - 1
    if num_params == 0:
        res = Results['OLS']
        return (None, res['RMSE'])
    elif num_params == 1:
        return get_best_one_param(Results, method)
    elif num_params == 2:
        return get_best_two_params(Results, method)
    elif num_params == 3:
        return get_best_three_params(Results, method)
    else:
        print("Something went horribly wrong")
    
    
def plot_cross_val(Results, show=True, save=True, save_file="cross_validation.png"):
    
    # TODO: this should be able to handle any of our models in RMSE
    
    K = Results['Elastic_Net']['RMSE'].shape[0]
    lines = []
    labels = []
    for model_name in Results.keys():
        _, model_rmse = get_best_rmse(Results, model_name)
        lines.append(model_rmse)
        labels.append(f"{model_name} ({model_rmse.mean().round(2)})")
        

    plt.plot(np.array(lines).T, label=labels)
    plt.xlabel("Folds")
    plt.xticks(np.arange(K))
    plt.ylabel("RMSE")
    plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1))
    
    if show:
        plt.show()
    if save:
        plt.savefig(f"figures/{save_file}")
        
        
def plot_method_cv(Results, method, param, show=True, save=True, save_file=None):
    if not save_file:
        save_file = f"cross_validation_{method}_{param}.png"
        
    res = Results[method]
    K = res['RMSE'].shape[0]
    rmse = np.zeros((K, len(res[param])))
    num_params = len(res.keys())-1
    param_idx = list(res.keys()).index(param)
    for k in range(K): 
        for i, md in enumerate(res[param]):
            rmse_for_fold = res["RMSE"][k]
            if num_params == 1:
                rmse_for_param = rmse_for_fold[i]
            elif num_params == 2:
                if param_idx == 0:
                    rmse_for_param = rmse_for_fold[i, :]
                else:
                    rmse_for_param = rmse_for_fold[:, i]
            elif num_params == 3:
                if param_idx == 0:
                    rmse_for_param = rmse_for_fold[i, :, :]
                elif param_idx== 1:
                    rmse_for_param = rmse_for_fold[:, i, :]
                else:
                    rmse_for_param = rmse_for_fold[:, :, i]
            else:
                print("too many params")


            rmse[k, i] = rmse_for_param.mean()
    plt.plot(res[param], rmse.T, label=np.arange(K)+1)
    #plt.xticks(np.arange(len(res[param])), )
    plt.xlabel(param)
    plt.ylabel("RMSE")
    plt.title(method)
    plt.legend(loc='upper right')

    if show:
        plt.show()
    if save:
        plt.savefig(f"figures/{save_file}")