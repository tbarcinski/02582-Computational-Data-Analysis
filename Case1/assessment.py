import pickle

import matplotlib.pyplot as plt
import numpy as np

def get_results(results_file):
    file = open(f"results/{results_file}", "rb") 
    RMSE = pickle.load(file)
    return RMSE


def plot_cross_val(RMSE, show=True, save=True, save_file="cross_validation.png"):
    
    # TODO: this should be able to handle any of our models in RMSE
    K = len(RMSE['OLS'])
    
    plt.bar(np.arange(K)-0.4, RMSE['OLS'], label='OLS', width=0.2)

    best_lambda_ridge_idx = RMSE['Ridge'].mean(axis=0).argmin()
    plt.bar(np.arange(K)-0.2, RMSE['Ridge'][:, best_lambda_ridge_idx], label=f'Ridge', width=0.2) #TODO: include best lambda in legend

    best_lambda_lasso_idx = RMSE['Lasso'].mean(axis=0).argmin()
    plt.bar(np.arange(K), RMSE['Lasso'][:, best_lambda_lasso_idx], label=f'Lasso', width=0.2) #TODO: include best lambda in legend

    plt.bar(np.arange(K)+0.2, RMSE['KNN'], label=f'KNN', width=0.2)
    plt.bar(np.arange(K)+0.4, RMSE['Weighted KNN'], label=f'Weighted KNN', width=0.2)

    plt.xlabel("Folds")
    plt.xticks(np.arange(K))
    plt.ylabel("RMSE")
    plt.legend(loc='lower right')
    
    
    if show:
        plt.show()
    if save:
        plt.savefig(f"figures/{save_file}")
        
        
def main():
    
    results_file = "results_rmse_2023-02-27 20:40:59.021065.pickle"
    RMSE = get_results(results_file)
    plot_cross_val(RMSE, show=False)
    
    
    
if __name__ == '__main__':
    main()

