# Import modules
import pandas as pd
from math import sqrt
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from joblib import dump

from utils import (iteration_train, get_final_model,
    compute_metrics, get_clf, get_model_name, perform_imputation)


#%% Creating Dirs
ROOT_PATH = './'
metrics_dir = os.path.join(ROOT_PATH, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)
dir_out = os.path.join(ROOT_PATH, 'Model_Group_06')
os.makedirs(dir_out, exist_ok=True)

#%% Seed value
# Apparently you may use different seed values at each stage
seed_value = 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# %% Load of the dataframe
path_to_dataset = os.path.join(ROOT_PATH, 'data','Full_Dataset.csv')
df = pd.read_csv(path_to_dataset)
print(df)

# summarize the number of rows with missing values for each column
percs = np.zeros(df.shape[1])

temp = df.columns.tolist()
for i in range(df.shape[1]):
    # count number of rows with missing values
    n_miss = df[temp[i]].isnull().sum()
    perc = n_miss / df.shape[0] * 100
    print('> {:3d}, Missing: {:4d} ({:5.1f}%)'.format(i, n_miss, perc))
    percs[i] = perc

for perc, col in zip(percs, temp):
    if (perc > 80):
        print("Perc = {:2f} Feat Name = {}".format(perc, col))

do_replace_negative = False
if do_replace_negative:
    cs = ['SAPS-I', 'SOFA']
    for c in cs:
        x = df[c]
        idx = x < 0
        df.loc[idx, c] = np.nan

# %%  Removal of features containing more than 80% of NaN
percentage = 0.2
print("Percentage (of not NaN) for keeping the column: ", percentage)
print("DataFrame shape before NaN removal:", df.shape)
thresh = round(df.shape[0] * percentage)
df = df.dropna(thresh=thresh, axis=1)

# %% Definition of X and y
X = df.drop(['recordid', 'In-hospital_death'], axis=1)
y = df['In-hospital_death']
X_feat_list = X.columns.tolist()

print("Dataframe with", X.shape[0], "samples. Minimum number of features:", format(sqrt(X.shape[0]), '.0f'))

# %%
target_count = y.value_counts()
print('Class 0 (control):', target_count[0])
print('Class 1 (septic):', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)')

# %%
y_test_all = y.to_numpy()
y_test_hat_l_all = np.zeros(y.shape)
y_test_hat_all = np.zeros(y.shape)
num_folders = 10

recalls    = np.zeros(num_folders)
precisions = np.zeros(num_folders)
auprc      = np.zeros(num_folders)
auroc      = np.zeros(num_folders)
thresholds = np.zeros(num_folders)

skf = StratifiedKFold(n_splits=num_folders, random_state=1001, shuffle=True)
X_np = X.to_numpy()
y_np = y.to_numpy()

# Use Scikit-Learn
clf = get_clf()
model_name = get_model_name(clf)
print("Model = {}".format(model_name))

for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    print("Crossvalidation Iter: {} / {}".format(idx+1, num_folders))
    X_train_np, X_test_np = X_np[train_index, :], X_np[test_index, :]
    X_train = pd.DataFrame(X_train_np, columns=X.columns)
    X_test = pd.DataFrame(X_test_np, columns=X.columns)
    y_train_np, y_test_np = y_np[train_index], y_np[test_index]
    y_train = pd.Series(y_train_np)
    y_test  = pd.Series(y_test_np)

    X_train, imputer = perform_imputation(X_train, imputer=None)
    X_test, _        = perform_imputation(X_test,  imputer=imputer)

    # Normalizing data
    scaler = MinMaxScaler()
    X_train_new = scaler.fit_transform(X_train)
    X_test_new = scaler.transform(X_test)

    clf = get_clf()

    X_colnames = X_train.columns
    clf, best_th_pr, ytrain_hat, ytrain_hat_l, ytest_hat, ytest_hat_l = \
        iteration_train(X_train_new, X_test_new, y_train, clf)

    thresholds[idx] = best_th_pr
    recall, precision, prc_auc, roc_auc = compute_metrics(y_test, ytest_hat, ytest_hat_l)
    recalls[idx] = recall
    precisions[idx] = precision
    auprc[idx] = prc_auc
    auroc[idx] = roc_auc

    y_test_hat_all[test_index] = ytest_hat
    y_test_hat_l_all[test_index] = ytest_hat_l


# Metrics computation
s1 = np.minimum(recalls, precisions)
print("----------------------------------")
print("Mean and Std")
print("Recall    = {:.3f} +/- {:.3f}".format(np.mean(recalls), np.std(recalls,ddof=1)))
print("Precision = {:.3f} +/- {:.3f}".format(np.mean(precisions), np.std(precisions,ddof=1)))
print("Min(R,P)  = {:.3f} +/- {:.3f}".format(np.mean(s1), np.std(s1,ddof=1)))
print("PRC AUC   = {:.3f} +/- {:.3f}".format(np.mean(auprc), np.std(auprc,ddof=1)))
print("ROC AUC   = {:.3f} +/- {:.3f}".format(np.mean(auroc), np.std(auroc,ddof=1)))
print("Threshold = {:.3f} +/- {:.3f}".format(np.mean(thresholds), np.std(thresholds,ddof=1)))
print("----------------------------------")

# Save Metrics
metrics = {
    'min' : s1,
    'auprc' : auprc,
    'auroc' : auroc
}
dump(metrics, os.path.join(metrics_dir,'metrics_{}.joblib'.format(model_name)))

# ---------------------------------#
#      Cross-validation Results    #
# ---------------------------------#
# Recall    = 0.519 +/- 0.051
# Precision = 0.523 +/- 0.018
# Min(R,P)  = 0.502 +/- 0.036
# PRC AUC   = 0.534 +/- 0.034
# ROC AUC   = 0.859 +/- 0.011

#%% Run on full dataset
model_name = "ensemble_5"

# Training the Imputer
X, imputer = perform_imputation(X)

# Training of the Scaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Feature Selection
X_new = X_normalized

X_colnames = X.columns

clf_final, best_th_pr, y_hat, y_hat_l = \
    get_final_model(X_new, y)

#%%
# Save data into files
do_save = input('Do you want to save trained model for final submission? (y/n)')
if do_save:
    dump(X_feat_list, os.path.join(dir_out, "featlist.joblib") )
    dump(scaler, os.path.join(dir_out, "scaler.joblib"))
    dump(clf_final, os.path.join(dir_out, "filename.joblib"))
    dump(best_th_pr, os.path.join(dir_out, "bestTHR.joblib"))
    dump(imputer, os.path.join(dir_out, "imputer.joblib"))


#%%
do_pi = input('Do you want to compute Permutation Importance? (y/n)')
if do_pi == 'y':
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(clf_final, random_state=1).fit(X_new, y)
    eli5.show_weights(perm, feature_names=X_new.columns.tolist())

    results_0 = perm.results_[0]

    results_mean = np.zeros(results_0.shape)
    results_std = np.std(results_0.shape)

    perm_means = np.mean(perm.results_, axis=0)
    perm_stds  = np.std (perm.results_, axis=0)

    results_0_copy = np.copy(perm_means)

    variable_to_show = 15

    importances_normalized = results_0_copy
    indices_sorted = np.argsort(importances_normalized)[::-1]
    importances_sorted = importances_normalized[indices_sorted]
    colnames_sorted = np.array(X.columns)[indices_sorted]
    errors_sorted = perm_stds[indices_sorted]

    importances_sorted_reduced = importances_sorted[:variable_to_show]
    colnames_sorted_reduced = colnames_sorted[:variable_to_show]
    errors_sorted_reduced = errors_sorted[:variable_to_show]

    f = plt.figure()
    plt.title("Ensemble(5) feature importance via permutation importance")
    plt.bar(range(variable_to_show), importances_sorted_reduced,
            yerr=errors_sorted_reduced, alpha=0.7, align='center')
    plt.ylabel("Score")
    plt.xticks(range(variable_to_show), colnames_sorted_reduced, rotation=90)
    plt.xlim([-1, variable_to_show])
    plt.ylim([0, 0.0225])
    plt.show()
    f.savefig("feature_importance.png")
