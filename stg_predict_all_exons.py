import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats
import torch
from stg import STG
import os
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error
from collections import Counter
import json
import copy

import numpy as np
import torch
import random
from sklearn.model_selection import KFold
import pickle

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

with open(r"ensg2gene_dict.pkl" ,'rb') as file:
    gene2ensg_dict = pickle.load(file)

full_path = "GTEx_V7_TPM_RBP_Full.csv"
sub_100k_path = "GTEx_V7_TPM_RBP_Sub_100K.csv"
sub_10k_path = "GTEx_V7_TPM_RBP_Sub_10K.csv"

params = {
    # "device": "cuda:1",
    "device": "cpu",
    "lasso_lam": np.logspace(-3,-0.5, 10),
    "stg_lam": np.logspace(-3, 0.3, 10),
    "output_dim": 1,
    "group_type": "prad",
    "min_num_feat": 60,
    "stg_epochs": 100,
    "batch_size": 128,
    "lr": 0.001,
    "hidden_dims": [500, 50, 50, 16],
    "activation": "relu",
    "print_interval": 20,
    "stg_thr": 0.1,
    "n_splits": 5,
}
feature_selection = True
mse_loss = torch.nn.MSELoss()



def get_group_k_fold(X_dict, params):
    folds = []
    seen_folds = set()
    gkf = GroupKFold(n_splits=params["n_splits"])
    
    for idx, (train_index, test_index) in enumerate(gkf.split(X_dict["X"], groups=X_dict["X"]["SUBJID"])):
        if len(folds) == params["n_splits"]:
            break
        train_ratio = len(train_index) / (len(train_index) + len(test_index))
        if train_ratio >= 0.50:
            train_tuple = tuple(train_index)
            test_tuple = tuple(test_index)
            fold_tuple = (train_tuple, test_tuple)
            if fold_tuple not in seen_folds:
                folds.append((train_index, test_index))
                seen_folds.add(fold_tuple)
    
    for i, (train_idxs, test_idxs) in enumerate(folds):
        print(f"Fold {i + 1}:")
        print(f"Train indices: {len(train_idxs)}")
        print(f"Test indices: {len(test_idxs)}")
        print("-" * 30)
    
    return folds

def get_predicted_exon_psi(X_dict, y, y_10K, y_100K, folds, lam, model):

    out_dict = {}

    for idx, (train_idxs, test_idxs) in enumerate(folds):

        X_train = X_dict["X"][train_idxs,:]
        y_train = y[train_idxs].reshape(-1, 1)

        X_test = X_dict["X"][test_idxs,:]
        X_test_100K = X_dict["X_100K"][test_idxs,:]
        X_test_10K = X_dict["X_10K"][test_idxs,:]
        y_test = y[test_idxs].reshape(-1, 1)

        y_test_10K = y_10K[test_idxs].reshape(-1, 1)
        y_test_100K = y_100K[test_idxs].reshape(-1, 1)
        
        mse_loss = torch.nn.MSELoss()

        if model == "stg":

            feature_selection = True

            model = STG(
                task_type="regression",
                input_dim=X_train[:, :].shape[1],
                output_dim=params["output_dim"],
                hidden_dims=params["hidden_dims"],
                activation=params["activation"],
                optimizer="Adam",
                learning_rate=params["lr"],
                batch_size=params["batch_size"],
                feature_selection=feature_selection,
                sigma=0.5,
                lam=lam,
                random_state=1,
                device=params["device"],
            )
            model.fit(
                X_train,
                y_train,
                nr_epochs=params["stg_epochs"],
                valid_X=X_test,
                valid_y=y_test,
                print_interval=params["print_interval"],
            )

            onez = np.ones(X_test.shape)
            onez[:, model.get_gates(mode="prob") < params["stg_thr"]] = 0
            num_selected_genes = np.sum(model.get_gates(mode="prob") > params["stg_thr"])

            # Full
            y_pred_full = model.predict(X_test * onez)

            # 100K
            y_pred_100K = model.predict(X_test_100K * onez)

            # 10K
            y_pred_10K = model.predict(X_test_10K * onez)

        else:
            clf = linear_model.Lasso(alpha=lam, max_iter=1000, tol=1e-1)
            clf.fit(X_train, y_train)

            num_selected_genes = np.where(np.abs(clf.coef_) > 0)[0].shape[0]
            y_pred_full = clf.predict(X_test)
            y_pred_100K = clf.predict(X_test_100K)
            y_pred_10K = clf.predict(X_test_10K)

        corr_full = np.corrcoef(y_test.reshape(-1), y_pred_full.reshape(-1))[0, 1]
        mse_full = mse_loss(torch.from_numpy(y_test).view(-1), torch.from_numpy(y_pred_full).view(-1)).item()

        corr_100K = np.corrcoef(y_test.reshape(-1), y_pred_100K.reshape(-1))[0, 1]
        mse_100K = mse_loss(torch.from_numpy(y_test).view(-1), torch.from_numpy(y_pred_100K).view(-1)).item()

        corr_10K = np.corrcoef(y_test.reshape(-1), y_pred_10K.reshape(-1))[0, 1]
        mse_10K = mse_loss(torch.from_numpy(y_test).view(-1), torch.from_numpy(y_pred_10K).view(-1)).item()

        out_dict[idx] = {
                         "y_pred_full": y_pred_full,
                         "y_pred_100K": y_pred_100K,
                         "y_pred_10K": y_pred_10K,

                         "y_test": y_test,
                         "y_test_10K": y_test_10K,
                         "y_test_100K": y_test_100K,
                         "num_selected_genes": num_selected_genes,

                         "corr_full": corr_full,
                         "corr_100K": corr_100K,
                         "corr_10K": corr_10K,

                         "mse_full": mse_full,
                         "mse_100K": mse_100K,
                         "mse_10K": mse_10K,

                         "test_idxs": test_idxs
                         }

    return out_dict

def run_all_exons(lam, folds, model):


    X_df = pd.read_csv(full_path)
    X_df_100K = pd.read_csv(sub_100k_path)
    X_df_10K = pd.read_csv(sub_10k_path)
    gene_start_idx = 4
    y_df = pd.read_csv(r"GTEx_V7_Sampled_Manual_PSI.csv")
    y_df_10K = pd.read_csv(r"GTEx_V7_SubSampled10K_Manual_PSI.csv")
    y_df_100K = pd.read_csv(r"GTEx_V7_SubSampled100K_Manual_PSI.csv")

    X_df["uniq"] = X_df["uniq"].apply(lambda x: x.split(".txt")[0])
    X_df_100K["uniq"] = X_df_100K["uniq"].apply(lambda x: x.split(".txt")[0])
    X_df_10K["uniq"] = X_df_10K["uniq"].apply(lambda x: x.split(".txt")[0])

    n_rows = 1
    last_n_rows = y_df.iloc[-n_rows:].reset_index(drop=True)
    y_df = y_df.iloc[:-n_rows].reset_index(drop=True)

    common_uniq_values = set(X_df_100K['uniq']).intersection(set(X_df_10K['uniq'])).intersection(set(X_df['uniq'])).intersection(set(y_df['uniq'])).intersection(set(y_df_10K['uniq'])).intersection(set(y_df_100K['uniq']))

    print(f"Number of common samples {len(common_uniq_values)}")

    X_df = X_df[X_df['uniq'].isin(common_uniq_values)]
    X_df_100K = X_df_100K[X_df_100K['uniq'].isin(common_uniq_values)]
    X_df_10K = X_df_10K[X_df_10K['uniq'].isin(common_uniq_values)]
    y_df = y_df[y_df['uniq'].isin(common_uniq_values)]
    y_df_10K = y_df_10K[y_df_10K['uniq'].isin(common_uniq_values)]
    y_df_100K = y_df_100K[y_df_100K['uniq'].isin(common_uniq_values)]

    X_df = X_df.sort_values(by=['uniq'])
    X_df_100K = X_df_100K.sort_values(by=['uniq'])
    X_df_10K = X_df_10K.sort_values(by=['uniq'])
    y_df = y_df.sort_values(by=['uniq'])
    y_df_10K = y_df_10K.sort_values(by=['uniq'])
    y_df_100K = y_df_100K.sort_values(by=['uniq'])

    y_df = pd.concat([y_df, last_n_rows], ignore_index=True)

    X_dict["X"] = X_df.iloc[:, gene_start_idx:].astype("double").to_numpy()
    X_dict["X_100K"] = X_df_100K.iloc[:, gene_start_idx:].astype("double").to_numpy()
    X_dict["X_10K"] = X_df_10K.iloc[:, gene_start_idx:].astype("double").to_numpy()

    for key in X_dict.keys():
        X_dict[key] = np.nan_to_num(X_dict[key], nan=0)
        X_dict[key] = stats.zscore(X_dict[key], axis=0)
        X_dict[key] = np.nan_to_num(X_dict[key], nan=0)

    all_exons_dict = {}
    for exon_idx, col in enumerate(y_df.columns[2:20]):
        X_dict_tmp = copy.deepcopy(X_dict)

        exon_name = col
        gene_name = y_df[col].iloc[-1]

        if gene_name in X_df.columns:
            gene_idx = np.where(X_df.columns == gene_name)[0] - gene_start_idx
            for key in X_dict_tmp.keys():
                X_dict_tmp[key] = np.delete(X_dict_tmp[key], gene_idx, axis=1)

        y = y_df[col].iloc[:-n_rows].astype("double").to_numpy().reshape(-1)
        y = 10*y - 2.5
        y = np.nan_to_num(y, 0)

        y_10K = y_df_10K[col].astype("double").to_numpy().reshape(-1)
        y_10K = np.nan_to_num(y_10K, 0)
        y_100K = y_df_100K[col].astype("double").to_numpy().reshape(-1)
        y_100K = np.nan_to_num(y_100K, 0)

        exon_dict = get_predicted_exon_psi(X_dict_tmp, y, y_10K, y_100K, folds, lam, model)
        all_exons_dict[exon_name] = exon_dict

        del X_dict_tmp

        if exon_idx == 5:
            break

    return all_exons_dict

X_df = pd.read_csv(full_path)
X_df_100K = pd.read_csv(sub_100k_path)
X_df_10K = pd.read_csv(sub_10k_path)
gene_start_idx = 4
y_df = pd.read_csv(r"GTEx_V7_Sampled_Manual_PSI.csv")
y_df_10K = pd.read_csv(r"GTEx_V7_SubSampled10K_Manual_PSI.csv")
y_df_100K = pd.read_csv(r"GTEx_V7_SubSampled100K_Manual_PSI.csv")

X_df["uniq"] = X_df["uniq"].apply(lambda x: x.split(".txt")[0])
X_df_100K["uniq"] = X_df_100K["uniq"].apply(lambda x: x.split(".txt")[0])
X_df_10K["uniq"] = X_df_10K["uniq"].apply(lambda x: x.split(".txt")[0])

common_uniq_values = set(X_df_100K['uniq']).intersection(set(X_df_10K['uniq'])).intersection(set(X_df['uniq'])).intersection(set(y_df['uniq'])).intersection(set(y_df_10K['uniq'])).intersection(set(y_df_100K['uniq']))

print(f"Number of common samples {len(common_uniq_values)}")

X_df = X_df[X_df['uniq'].isin(common_uniq_values)].reset_index(drop=True, inplace=False)
X_df_100K = X_df_100K[X_df_100K['uniq'].isin(common_uniq_values)].reset_index(drop=True, inplace=False)
X_df_10K = X_df_10K[X_df_10K['uniq'].isin(common_uniq_values)].reset_index(drop=True, inplace=False)
y_df = y_df[y_df['uniq'].isin(common_uniq_values)].reset_index(drop=True, inplace=False)
y_df_10K = y_df_10K[y_df_10K['uniq'].isin(common_uniq_values)].reset_index(drop=True, inplace=False)
y_df_100K = y_df_100K[y_df_100K['uniq'].isin(common_uniq_values)].reset_index(drop=True, inplace=False)

X_df = X_df.sort_values(by=['uniq'])
X_df_100K = X_df_100K.sort_values(by=['uniq'])
X_df_10K = X_df_10K.sort_values(by=['uniq'])
y_df = y_df.sort_values(by=['uniq'])
y_df_10K = y_df_10K.sort_values(by=['uniq'])
y_df_100K = y_df_100K.sort_values(by=['uniq'])

X_dict = {"X": X_df, "X_100K": X_df_100K, "X_10K": X_df_10K}
folds = get_group_k_fold(X_dict, params)

all_results = {}
for model in ["stg", "lasso"]: 
    
    for lam_idx, lam in enumerate(params[f"{model}_lam"]):
        all_exons_dict = run_all_exons(lam, folds, model)
        all_results[lam_idx] = all_exons_dict
    with open(fr"{model}_results.pkl" ,'wb') as file:
        pickle.dump(all_results, file)

