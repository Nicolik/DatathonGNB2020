def testing_function(df):
    """
    Testing Function Developed by Group 06
    :param df: Dataframe, with the same format of the "Full_Dataset.csv" provided by the organizers
    :return: predicted record ids, probabilities and thresholded predictions
    """
    # Libraries import
    from joblib import load
    import os
    import pandas as pd

    # Eventually change this to proper path
    subm_path = 'Model_Group_06'

    start_feat = load(os.path.join(subm_path,'featlist.joblib'))
    scaler     = load(os.path.join(subm_path,'scaler.joblib'))
    imputer    = load(os.path.join(subm_path, 'imputer.joblib'))
    ensemble   = load(os.path.join(subm_path,'filename.joblib'))
    thresh     = load(os.path.join(subm_path,'bestTHR.joblib'))

    # Test on Dataset_C
    feat_sub = start_feat + ['recordid', 'In-hospital_death']
    DM_C = df[feat_sub]

    # Definition of X subset
    X_C = DM_C[start_feat]
    recordid = DM_C['recordid']

    # Imputer
    np_array = imputer.transform(X_C)
    X_C = pd.DataFrame(np_array, columns=start_feat)

    # Min Max Normalization
    X_C_normalized = scaler.transform(X_C)

    # Probabilities predicted by the ensemble
    y_C_prob = ensemble.predict_proba(X_C_normalized)

    # Consider only the probabilities for the positive class
    y_C_prob_pos = y_C_prob[:,1]

    # Thresholded final results
    y_C_pred = y_C_prob_pos >= thresh

    return recordid, y_C_prob_pos, y_C_pred
