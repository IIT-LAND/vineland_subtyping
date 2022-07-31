from sklearn.preprocessing import StandardScaler
from umap import UMAP
import pickle as pkl
import os
import strat.utils as ut
import logging
#-------------------------------------------------------------------------------------------------------------------------
#### what did I modify?
#### y_TR = 'cluster_domain' 
#### take first observation
### nb_ is clusters only on those who have a longitudinal assessment (if only T1 present, they will be ecluded)
# ------------------------------------------------------------------------------------------------------------------------
ndar_feat = [ 'communicationdomain_totalb',
             'livingskillsdomain_totalb',
             'socializationdomain_totalb',
             'motorskillsdomain_totalb']### add motor

ucsd_feat = ['vine_ComTotal_DomStd',
             'vine_DlyTotal_DomStd',
             'vine_SocTotal_DomStd',
             'vine_MtrTotal_DomStd']### add motor


def prepare_ucsd(ndar, ucsd_long, period='P1'):
    """
    Function that prepares training and test sets (only selecting the features of interest)
    that are to be input to `predict_labels`.

    :param ndar: dataframe from NDAR
    :param ucsd_long: names of ucsd long-format file
    :param period: P1 or P2 if we want to apply the trained classifier from NDAR P1 or P2
    :return: data frame, series, dataframe
    """
    ucsd_long = ucsd_long.loc[ucsd_long[ucsd_feat].isnull().astype(int).apply(sum, axis=1) == 0]
    ucsd_long = ucsd_long.drop_duplicates().copy()
    gui_rm = ucsd_long.reset_index().duplicated('subjectid', keep=False)
    gui_rm.index = ucsd_long.index
    ucsd_long = ucsd_long.loc[gui_rm]

    X_tr = ndar[[c for c in ndar_feat]]
    y_tr = ndar['cluster_domain'] ## take cluster domain only!!!

    if period == 'P1':
        X_ts = ucsd_long.loc[ucsd_long.vine_agemo <= 72]
        X_ts = X_ts.loc[X_ts.vine_agemo >= 12]
    elif period == 'P2':
        X_ts = ucsd_long.loc[ucsd_long.vine_agemo > 72]
        X_ts = X_ts.loc[X_ts.vine_agemo <= 156]
    X_ts.reset_index(inplace=True)
    X_ts = X_ts.sort_values(['subjectid', 'vine_agemo'])

    X_ts.drop_duplicates(['subjectid'], keep='first', inplace=True)
    X_ts.index = X_ts.subjectid
    logging.info(f' Selected subjects at their first encounter done at age included in '
                 f'period {period}. Mean (sd) interview age is {round(X_ts.vine_agemo.describe()["mean"], 2)} '
                 f'({round(X_ts.vine_agemo.describe()["std"], 2)}) '
                 f'-- Min / Max:[{round(X_ts.vine_agemo.describe()["min"], 2)}; '
                 f'{round(X_ts.vine_agemo.describe()["max"], 2)}]')
    X_ts.drop('subjectid', axis=1, inplace=True)
    return X_tr, y_tr, X_ts[ucsd_feat], ucsd_long

# def predict_labels(X_tr, X_ts, y_tr, classifier):
#     """
#     This function trains a classifier on a training set given the labels
#     and applies it to a cross-sectional test set in order to obtain new
#     labelings. It returns a dictionary with the label found for each subject (key).
#     Trained model is saved in the output folder.

#     :param X_tr: training set
#     :type X_tr: pandas dataframe
#     :param X_ts: test set
#     :type X_ts: pandas dataframe
#     :param y_tr: labels for training set
#     :type y_tr: pandas dataframe
#     :param classifier: classifier
#     :type classifier: method
#     :return: dictionary
#     """
#     scaler = StandardScaler()
#     umap = UMAP(n_neighbors=30, min_dist=0.0, random_state=42)

#     subj = X_ts.index

#     X_tr = umap.fit_transform(scaler.fit_transform(X_tr))
#     model_fit = classifier.fit(X_tr, y_tr)
#     pkl.dump(model_fit, open(os.path.join(ut.out_folder,
#                                           'fittedmodel_NDARts.sav'), 'wb'))

#     X_ts = umap.transform(scaler.transform(X_ts))
#     pred_labels = model_fit.predict(X_ts)

#     label_dict = {s: lab for s, lab in zip(subj, pred_labels)}
#     return label_dict

# from isotta's code modify this function to have in this notebook the outputs :scaler  and umap
# this is the function to train the classifier (including the scaling and umap preprocessing)
def predict_labels(X_tr, X_ts, y_tr, classifier):
    """ 
    This function trains a classifier on a training set given the labels
    and applies it to a cross-sectional test set in order to obtain new
    labelings. It returns a dictionary with the label found for each subject (key).
    Trained model is saved in the output folder.

    :param X_tr: training set
    :type X_tr: pandas dataframe
    :param X_ts: test set
    :type X_ts: pandas dataframe
    :param y_tr: labels for training set
    :type y_tr: pandas dataframe
    :param classifier: classifier
    :type classifier: method
    :return: dictionary
    """
    scaler = StandardScaler()
    umap = UMAP(n_neighbors=30, min_dist=0.0, random_state=42)

    subj = X_ts.index
    
    # do scaling and UMAP
    X_train_scaled = umap.fit_transform(scaler.fit_transform(X_tr))
    
    # save scaler and umap too
    scaler_2save = scaler.fit(X_tr)
    pkl.dump(scaler_2save, open(os.path.join(ut.out_folder,
                                          'scalerDomainONLY_NDARts.sav'), 'wb')) # save the trained scaler
    umap_2save = umap.fit(scaler.fit_transform(X_tr))
    pkl.dump(umap_2save, open(os.path.join(ut.out_folder,
                                          'umapDomainONLY_NDARts.sav'), 'wb')) #save the trained UMAP
    
    # TRAIN the model
    model_fit = classifier.fit(X_train_scaled, y_tr)
    
    #save the model
    pkl.dump(model_fit, open(os.path.join(ut.out_folder,
                                          'fittedmodelDomainONLY_NDARts.sav'), 'wb')) #save the trained classifier
   

    X_ts = umap.transform(scaler.transform(X_ts))
    pred_labels = model_fit.predict(X_ts)

    label_dict = {s: lab for s, lab in zip(subj, pred_labels)}
    return label_dict, model_fit, scaler, umap



### this function in modified by Vero in order to eliminate the UMAP preprocessing

def predict_labels_no_umap(X_tr, X_ts, y_tr, classifier):
    """ 
    This function trains a classifier on a training set given the labels
    and applies it to a cross-sectional test set in order to obtain new
    labelings. It returns a dictionary with the label found for each subject (key).
    Trained model is saved in the output folder.

    :param X_tr: training set
    :type X_tr: pandas dataframe
    :param X_ts: test set
    :type X_ts: pandas dataframe
    :param y_tr: labels for training set
    :type y_tr: pandas dataframe
    :param classifier: classifier
    :type classifier: method
    :return: dictionary
    """
    scaler = StandardScaler()
    #umap = UMAP(n_neighbors=30, min_dist=0.0, random_state=42)

    subj = X_ts.index
   
    X_train_scaled = scaler.fit_transform(X_tr)
    
    # save scaler and umap too 
    
    scaler_2save = scaler.fit(X_tr)
    pkl.dump(scaler_2save, open(os.path.join(ut.out_folder,
                                          'scalerDomainONLY_NDARts.sav'), 'wb')) # save the trained scaler
    #umap_2save = umap.fit(scaler.fit_transform(X_tr))
    #pkl.dump(umap_2save, open(os.path.join(ut.out_folder,
    #                                     'umapDomainONLY_NDARts.sav'), 'wb')) #save the trained UMAP
    
    model_fit = classifier.fit(X_train_scaled, y_tr)
    
    #save the model
    pkl.dump(model_fit, open(os.path.join(ut.out_folder,
                                          'fittedmodelDomainONLY_NDARts.sav'), 'wb')) #save the trained classifier
   

    X_ts = scaler.transform(X_ts)
    pred_labels = model_fit.predict(X_ts)

    label_dict = {s: lab for s, lab in zip(subj, pred_labels)}
    return label_dict, model_fit, scaler

def create_new(data, label_dict):
    """
    Returns a long-format longitudinal dataset with new labels, ordered according to subject IDs and
    interview period (1 to 5 timestamps).
    :param data: longitudinal dataset in long format
    :type data: pandas dataframe
    :param label_dict: dictionary with new labels as returned by `strat.ucsd_dataset.predict_labels`
    :type label_dict: dictionary
    :return: pandas dataframe
    """
    labels_vect = []
    drop_subj = []
    data = data.reset_index()
    for idx, row in data.iterrows():
        if row.subjectid in label_dict.keys():
            labels_vect.append(label_dict[row.subjectid])
        else:
            drop_subj.append(idx)
    data.drop(drop_subj, axis=0, inplace=True)
    data['cluster'] = labels_vect

    data = data.sort_values(['subjectid', 'time'])
    data.index = data.subjectid
    data.drop('subjectid', axis=1, inplace=True)
    return data
