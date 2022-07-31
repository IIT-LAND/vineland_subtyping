#import strat.utils as ut
import strat.utils as ut
import os
import pickle as pkl
from strat.import_data import ReadData
from strat.create_levels import vineland_levels
import logging
import numpy as np
import pandas as pd
import math


def dataset(ins_name_dict, pheno_file_name, ptype, save=True, phenotype=None,ids_2_remove=[9]):
    """
    This function takes as input a dictionary with the names of the files to consider (only one
    instrument is allowed) and the phenotype of the subjects to select (only None or "autism" is allowed).
     executes all the steps that are required in order to:

    The names of the instrument tables can be reported in utils.py
    under instrument_dict variable. The output are saved as pickle objects. The
    output folder where to store the returned data should be stated in utils.py (output_folder
    variable).

    :param ins_name_dict: dictionary with instrument table names
    :type ins_name_dict: dict
    :param pheno_file_name: name of the file with phenotypes
    :type pheno_file_name: str
    :param phenotype: select a phenotype, defaults to None
    :type phenotype: str
    :param ptype: type of interview period definition
    :type ptype: int
    :param save: save longitudinal information, defaults to True
    :type save: bool
    :return: merged longitudinal dataset (ordered wrt to subjectkey and interview_age), subject demographic info
    :rtype: pandas dataframe, dictionary of namedtuple
    :ids_2_remove = id to be revoved
    """
    data = ReadData(ins_name_dict, pheno_file_name, phenotype=phenotype)
    instrument_dict, demodict = data.data_wrangling(save)

    # For now only vineland dataset is enabled
    df = vineland_levels(instrument_dict,
                         ut.fixed_col,
                         ut.vineland_col_names,
                         ut.vineland_mergecol,
                         ptype=ptype)
    old_dim = df.shape
    ##################################################################################################
    df = df.loc[~df.collection_id.isin(ids_2_remove)]  # Discard UCSD data = 9
    ##################################################################################################
    logging.info(f"Dropped {old_dim[0] - df.shape[0]} subjects from UCSD longitudinal study")
    logging.info(f'Current number of observation: {df.shape[0]}\n\n')
    new_demodict = {k: demodict[k] for k in df.index}
    df['sex'] = [new_demodict[gui].sex for gui in df.index]
    if save:
        # Save longitudinal datasets
        pkl.dump((df, new_demodict),
                 open(os.path.join(ut.out_folder, 'longitudinal_data.pkl'), 'wb'))

    return df, new_demodict


def prepare_imputation(df, missing_perc=0.35):
    """
    This function drops subjects that report a percentage of missing information
    greater than `missing_perc` computed on all Vineland subdomains and domains.
    Written subdomain is dropped.

    :param df: instrument dataframe
    :type df: pandas dataframe
    :param missing_perc: missing information threshold
    :type missing_perc: float
    :return: modified instrument dataset
    :rtype: pandas dataframe
    """
    # df_rid = df.drop(['written_vscore'], axis=1).copy()
    df_rid = df.copy()
    d_feat, d_subj = _check_na_perc(df_rid)
    df_rid.insert(0, 'hasna', [math.ceil(val) for val in d_subj.values()])
    df_rid.insert(0, 'countna', list(d_subj.values()))
    ############################# here there is the problem of age bins ###############################################
    for sub in range(0,df_rid.shape[0]+1,1):
        if  df_rid.loc[sub,'interview_age'] < 50:
            df_rid.loc[sub,'ageyrs_round'] = round(df_rid.interview_age/12)
        else:
            df_rid.loc[sub,'ageyrs_round'] = 50
    logging.info(f'Percentage of missing information for each feature:\n'
                 f'{d_feat}\n'
                 f'Median - Min/Max percentage of missing information per subject:'
                 f'{np.median(list(d_subj.values()))} - {min(d_subj.values())}/{max(d_subj.values())}')
    logging.info(f'Threshold set at: {missing_perc}')
    df_rid.reset_index(inplace=True)
    idxtodrop = [idx for idx in df_rid.index.tolist() if d_subj[idx] > missing_perc]
    df_rid.drop(idxtodrop, axis=0, inplace=True)
    df_rid.sort_values(['subjectkey', 'interview_period'], inplace=True)
    df_rid.index = df_rid.subjectkey
    df_rid.drop(['subjectkey'], axis=1, inplace=True)
    logging.info(f'Dropped {len(idxtodrop)}')
    return df_rid


"""
Private functions
"""


def _check_na_perc(df):
    """
    Function that reports the maximum percentage of missing information
    and the percentage of missing information
    feature-wise (i.e., per each subject) and subject-wise (i.e., per each feature).
    Since subject can be repeated, the percentage of missing information feature-wise is
    reported for each index position.

    :param df: instrument dataframe
    :type df: pandas dataframe
    :return: dictionary of % of missing info for each feature,
        dictionary of % of missing info for each subject
    :rtype: dict, dict
    """
    nobs = df.shape[0]
    nfeat = len(df.columns.intersection(ut.vineland_col_names[1:]).tolist())
    cols = df.columns.intersection(ut.vineland_col_names[1:]).tolist()
    # Subject-wise
    na_feat = {col: (df[[col]].isna().astype(int).sum().tolist()[0] / nobs)
               for col in cols}
    # Feat-wise
    df_cp = df.reset_index().copy()
    na_subj = {int(idx): (df_cp[cols].iloc[idx].isna().astype(int).sum() / nfeat)
               for idx in df_cp.index.tolist()}
    return na_feat, na_subj


def _impute(train_df, test_df, model):
    """
    Function that perform missing data imputation
    on both train and test for a unique interview period.

    :param train_df: training set (all feature names except relationship)
    :type train_df: pandas dataframe
    :param test_df: test set (all feature names except relationship)
    :type test_df: pandas dataframe
    :param model: imputation method
    :type model: class
    :return: imputed training set, imputed test set
    :rtype: pandas dataframe, pandas dataframe
    """
    col_n = [nc for nc in train_df.columns.intersection(ut.vineland_col_names[1:])]
    tmp_tr = pd.DataFrame(model.fit_transform(train_df[col_n]), columns=col_n)
    tmp_ts = pd.DataFrame(model.transform(test_df[col_n]), columns=col_n)
    tmp_tr.index = train_df.index
    tmp_ts.index = test_df.index
    return tmp_tr, tmp_ts
