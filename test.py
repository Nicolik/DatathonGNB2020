#######################
# GROUP 06 SUBMISSION #
#######################

import pandas as pd
import os
from Model_Group_06.model import testing_function
from utils import compute_metrics, show_metrics

ROOT_PATH = '.'

#%% Train Dataframe
print("---------------------------")
print("# Train Dataset Dataframe #")
print("---------------------------")
path_to_dataset = os.path.join(ROOT_PATH, 'data', 'Full_Dataset.csv')
df = pd.read_csv(path_to_dataset)

recordid, y_C_prob, y_C_pred = testing_function(df)
y_test = df['In-hospital_death']
recall, precision, prc_auc, roc_auc = compute_metrics(y_test, y_C_prob, y_C_pred)
show_metrics(precision, recall, prc_auc, roc_auc)

# --------------------------#
#  Train Dataset Dataframe  #
# --------------------------#
# Precision  : 0.667
# Recall     : 0.663
# Min(P,R)   : 0.663
# AUPRC      : 0.745
# AUROC      : 0.950

#%% Test Dataframe
print("---------------------------")
print("# Test  Dataset Dataframe #")
print("---------------------------")
path_to_dataset = os.path.join(ROOT_PATH, 'data', 'Testing_Dataset.csv')
df = pd.read_csv(path_to_dataset)

recordid, y_C_prob, y_C_pred = testing_function(df)
y_test = df['In-hospital_death']
recall, precision, prc_auc, roc_auc = compute_metrics(y_test, y_C_prob, y_C_pred)
show_metrics(precision, recall, prc_auc, roc_auc)

# --------------------------#
#  Test  Dataset Dataframe  #
# --------------------------#
# Precision  : 0.565
# Recall     : 0.567
# Min(P,R)   : 0.565
# AUPRC      : 0.550
# AUROC      : 0.866
