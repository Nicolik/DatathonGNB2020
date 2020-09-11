import pandas as pd
from math import sqrt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def find_best_th_roc(ytrain, temp_ytrain_hat):
    fpr_train, tpr_train, th_roc_train = roc_curve(ytrain, temp_ytrain_hat)
    best_point_roc_x = np.array([0] * fpr_train.shape[0])
    best_point_roc_y = np.array([1] * tpr_train.shape[0])
    temp_x = (fpr_train - best_point_roc_x)
    temp_y = (tpr_train - best_point_roc_y)
    temp_sqrt = np.sqrt(np.square(temp_x) + np.square(temp_y))
    index_min_temp_sqrt = np.argmin(temp_sqrt)
    best_th_roc = th_roc_train[index_min_temp_sqrt]
    return best_th_roc


def find_best_th_min(y, y_hat):

    ths = np.linspace(0.001, 0.999, 198)

    recalls    = np.zeros(ths.shape)
    precisions = np.zeros(ths.shape)

    min_to_beat = 0
    th_to_beat  = 0
    pre_to_beat = 0
    rec_to_beat = 0
    best_index  = 0
    for i, th in enumerate(ths):
        y_hat_l = y_hat >= th
        CM = confusion_matrix(y, y_hat_l)
        (tn, fp, fn, tp) = CM.ravel()
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        recall = tp / (tp + fn)
        min_pr = min(precision, recall)
        if min_pr > min_to_beat:
            min_to_beat = min_pr
            th_to_beat = th
            pre_to_beat = precision
            rec_to_beat = recall
            best_index = i
    print("Best Threshold = {:.3f}".format(th_to_beat))
    print("[Precision = {:.3f} Recall = {:.3f}]".format(pre_to_beat, rec_to_beat))

    diff_pre_rec = abs(pre_to_beat - rec_to_beat)
    if diff_pre_rec > 0.1:
        print("Recalls    = ", recalls[best_index-10:best_index+10])
        print("Precisions = ", recalls[best_index-10:best_index+10])
        print("Thresholds = ", ths[best_index-10:best_index+10])
    return th_to_beat


def get_X_Y(df: pd.DataFrame):
    # Definition of X and y
    X = df.drop(['recordid', 'In-hospital_death'], axis=1)
    y = df['In-hospital_death']

    print("Dataframe with", X.shape[0], "samples. Minimum number of features:", format(sqrt(X.shape[0]), '.0f'))
    return X,y


def show_metrics(precision, recall, prc_auc, roc_auc):
    print("{:10s} : {:.3f}".format("Precision", precision))
    print("{:10s} : {:.3f}".format("Recall", recall))
    print("{:10s} : {:.3f}".format("Min(P,R)", min(precision, recall)))
    print("{:10s} : {:.3f}".format("AUPRC", prc_auc))
    print("{:10s} : {:.3f}".format("AUROC", roc_auc))


def compute_metrics(y_test_all, y_test_hat_all, y_test_hat_l_all):
    CM = confusion_matrix(y_test_all, y_test_hat_l_all)
    (tn, fp, fn, tp) = CM.ravel()
    fpr, tpr, th_roc = roc_curve(y_test_all, y_test_hat_all)
    roc_auc   = auc(fpr, tpr)
    pre, rec, th_prc = precision_recall_curve(y_test_all, y_test_hat_all)
    prc_auc   = auc(rec, pre)
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp)
    return recall, precision, prc_auc, roc_auc


def get_final_model(X, y, clf=None):
    if clf is None:
        clf = get_clf()

    # Fit the model
    clf.fit(X, y)

    # Compute predictions with best threshold
    y_hat = clf.predict_proba(X)
    y_hat = y_hat[:, 1]

    # Compute best threshold
    temp_y_hat = np.copy(y_hat)
    # threshold computed on training data
    best_th_pr = find_best_th_min(y, temp_y_hat)
    y_hat_l = (y_hat >= best_th_pr).astype(int)
    print("---------------------------")
    print("------- Final Model -------")
    print("---------------------------")
    return clf, best_th_pr, y_hat, y_hat_l


def get_ensemble_5():
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=50)
    clf3 = AdaBoostClassifier(n_estimators=50)
    clf4 = CalibratedClassifierCV(SVC(kernel='rbf'))
    clf5 = MLPClassifier(solver='adam', alpha=1e-5,
                         hidden_layer_sizes=(10), random_state=1)
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                         ('abc', clf3), ('svm', clf4), ('mlp', clf5)],
                             voting='soft', weights=[2,1,1,2,2])
    return eclf1


def get_ensemble():
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=50)
    clf3 = AdaBoostClassifier(n_estimators=50)
    clf4 = CalibratedClassifierCV(SVC(kernel='rbf'))
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                         ('abc', clf3), ('svm', clf4)],
                             voting='soft', weights=[2,1,1,2])
    return eclf1


def get_clf():
    # Logistic Regression
    # clf = LogisticRegression(max_iter=1000)

    # SVM RBF
    # svm = SVC(kernel='rbf')
    # clf = CalibratedClassifierCV(svm)

    # Random Forests
    # rf = RandomForestClassifier(n_estimators=50)
    # clf = CalibratedClassifierCV(rf)

    # AdaBoost Classifier
    # abc = AdaBoostClassifier(n_estimators=50)
    # clf = CalibratedClassifierCV(abc)

    # MLP
    # clf = MLPClassifier(solver='adam', alpha=1e-5,
    #                     hidden_layer_sizes=(10), random_state = 1)

    # Ensemble (Final Submission)
    # clf = get_ensemble()

    clf = get_ensemble_5()

    return clf


def iteration_train(X_train_new, X_test_new, y_train, clf):

    # Fit the model
    clf.fit(X_train_new, y_train)

    # Compute probabilities on test set
    ytest_hat = clf.predict_proba(X_test_new)
    ytest_hat = ytest_hat[:, 1]

    # Compute predictions with best threshold
    ytrain = y_train.to_numpy()
    ytrain_hat = clf.predict_proba(X_train_new)
    ytrain_hat = ytrain_hat[:, 1]

    # Compute best threshold
    temp_ytrain_hat = np.copy(ytrain_hat)
    best_th_pr = find_best_th_min(ytrain, temp_ytrain_hat)  # threshold computed on training data
    ytest_hat_l = (ytest_hat >= best_th_pr).astype(int)

    ytrain_hat_l = (ytrain_hat >= best_th_pr).astype(int)
    return clf, best_th_pr, ytrain_hat, ytrain_hat_l, ytest_hat, ytest_hat_l


def get_model_name(clf):
    model_name = clf.__str__()[:10]
    if 'Calibrated' in model_name:
        model_name += "_{}".format(clf.base_estimator)[:6]
    return model_name


def perform_imputation(X, imputer=None):
    X_feat_list = X.columns
    if imputer is None:
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        imputer.fit(X)
    np_array = imputer.transform(X)
    X = pd.DataFrame(np_array, columns=X_feat_list)
    return X, imputer
