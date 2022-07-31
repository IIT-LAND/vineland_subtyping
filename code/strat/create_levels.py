import pandas as pd
import logging
import re
import strat.utils as ut


def vineland_levels(vineland_dict, fixed_cols, col_names, mergecol, ptype=1):
    """
    It creates a unique-level dataframe merging different versions of Vineland instrument,
    namely Vineland II (Parent and Caregiver rating), Vineland II (rating form),
    and Vineland 3.

    :param vineland_dict: dataframes (multiple instrument versions for vineland)
    :type vineland_dict: dictionary
    :param fixed_cols: columns to select, in common to all NDAR datasets
    :type fixed_cols: list
    :param col_names: columns to select with adaptive behavior scores
    :type col_names: list
    :param mergecol: columns to merge are displayed as keys and values.
        Keys become the new column name
    :type mergecol: dictionary
    :param ptype: type of interview period bins, defaults to 1, i.e., 5 periods
    :type ptype: int
    :return: longitudinal (when available) Vineland dataset that includes all versions.
        Rows are sorted according to subjectkeys and interview age.
    rtype: pandas dataframe
    """
    # When separate columns are filled-in based on sex we merge them into one
    # Rename columns to uniform features and enable merging of the dataframes
    logging.info('Processing VINELAND')
    rid_dict = {}
    for k, df in vineland_dict.items():
        for c1, c2 in mergecol.items():
            if c1 in df.columns and c2 in df.columns:
                df.loc[:, c1] = df.loc[:, c1].fillna(df.loc[:, c2])
                df.drop(c2, axis=1, inplace=True)
            else:
                df.rename(columns={c2: c1}, inplace=True)
        rid_dict[k] = df[fixed_cols
                         + list(df.columns.intersection(col_names))]
    df_concat = pd.concat([rid_dict[k] for k in rid_dict.keys()],
                          sort=False)
    # Replace with NAs entries that are not conform
    for cn in df_concat.columns.intersection(col_names):
        if cn not in fixed_cols:
            if re.search('total', cn):
                df_concat.loc[df_concat[cn] > 140, cn] = None  # 999=NA/missing
                df_concat.loc[df_concat[cn] <= 0, cn] = None  # -999=NA/missing
            elif cn == 'relationship':
                try:
                    df_concat.loc[df_concat[cn] == 27, cn] = None  # 27=Missing data
                    df_concat.loc[df_concat[cn] <= 0, cn] = None  # -999=NA/Missing data
                except TypeError:
                    pass
            elif re.search('subdom', cn):
                df_concat.loc[df_concat[cn] == 999, cn] = None  # 999=NA/missing
                df_concat.loc[df_concat[cn] <= 0, cn] = None  # -999=NA/missing
                df_concat.loc[df_concat[cn] > ut.vinesubdom_thrs[cn]] = None  # subdomain scores not in range
    # Drop 2 subjects (see utils file)
    # that have wrong entries
    # df_concat.drop(ut.drop_subj_vine, axis=0, inplace=True)
    cnrow = df_concat.shape[0]  # store current number of subjects
    # Drop subjects with completely missing information
    df_concat.reset_index(inplace=True)
    drop_obs = _drop_obs(df_concat[df_concat.columns.intersection(col_names[1:])])
    df_concat.drop(drop_obs, axis=0, inplace=True)
    logging.info(f'Dropped {cnrow - df_concat.shape[0]} '
                 f'observations with completely missing information')
    cnrow = df_concat.shape[0]
    # Drop duplicates. All entries displaying the same interview age,
    # caretaker to which it was administered, and scores
    df_concat.drop_duplicates(['subjectkey',
                               'interview_age',
                               'relationship'] +
                              list(df_concat.columns.intersection(col_names)),
                              keep='last', inplace=True)
    df_concat.sort_values(by=['subjectkey',
                              'interview_age'],
                          axis=0, inplace=True)
    df_concat.index = df_concat['subjectkey']
    df_concat.drop('subjectkey', axis=1, inplace=True)
    ###########################################################################################################################
    df_concat.insert(0, 'interview_period', _generate_age_bins_mod(df_concat.interview_age, ptype)) #### here is generate bin mod inserted########################################################################################## for older subjects class
    logging.info(f'Dropped {cnrow - df_concat.shape[0]} duplicated observations')
    return df_concat


def _drop_obs(df):
    """
    Function that takes as input a dataframe (only the desired features)
    and returns a list of subjects to drop.
    Subjects to drop are those for which all entries are either None or 0.

    :param df: dataset with desired features
    :type df: pandas dataframe
    :return: list of subjects to drop
    :rtype: list
    """
    drop_subj = []
    for sid, row in df.iterrows():
        try:
            if sum(row.dropna()) == 0:
                drop_subj.append(sid)
            elif row.count() == 0:
                drop_subj.append(sid)
        except TypeError:  # compatible to ados
            if sum(row.astype(float).dropna()) == 0:
                drop_subj.append(sid)
            elif row.astype(float).count() == 0:
                drop_subj.append(sid)
    return drop_subj


def _generate_age_bins(interview_age, ptype=1):
    """
    Returns the time period from the age of assessment. Different possibilities are available.
    type = 1: 5 age bins at 30, 72, 156, 204 months
    type = 2: 3 age bins at 72, 156 months

    :param interview_age: interview age and subject keys as index
    :type interview_age: pandas Series
    :param ptype: desired interview periods, defaults to 1
    :type ptype: int
    :return: vector of period strings
    :rtype: list
    """
    age_bins = []
    if ptype == 1:
        for aoa in interview_age:
            if 0 < float(aoa) <= 30.0:
                age_bins.append('P1')
            elif 30.0 < float(aoa) <= 72.0:
                age_bins.append('P2')
            elif 72.0 < float(aoa) <= 156.0:
                age_bins.append('P3')
            elif 156.0 < float(aoa) < 204.0:
                age_bins.append('P4')
            else:
                age_bins.append('P5')
    else:
        for aoa in interview_age:
            if 12 <= float(aoa) <= 72.0:
                age_bins.append('P1')
            elif 72.0 < float(aoa) <= 156.0:
                age_bins.append('P2')
            else:
                age_bins.append('oth')
    return pd.Series(age_bins, index=interview_age.index)


def _generate_age_bins_mod(interview_age, ptype=1):
    """
    Returns the time period from the age of assessment. Different possibilities are available.
    type = 1: 5 age bins at 30, 72, 156, 204 months
    type = 2: 3 age bins at 72, 156 months

    :param interview_age: interview age and subject keys as index
    :type interview_age: pandas Series
    :param ptype: desired interview periods, defaults to 1
    :type ptype: int
    :return: vector of period strings
    :rtype: list
    """
    age_bins = []
    if ptype == 1:
        for aoa in interview_age:
            if 0 < float(aoa) <= 30.0:
                age_bins.append('P1')
            elif 30.0 < float(aoa) <= 72.0:
                age_bins.append('P2')
            elif 72.0 < float(aoa) <= 156.0:
                age_bins.append('P3')
            elif 156.0 < float(aoa) < 204.0:
                age_bins.append('P4')
            else:
                age_bins.append('P5')
    else:
        for aoa in interview_age:
            if 12 <= float(aoa) <= 72.0:
                age_bins.append('P1')
            elif float(aoa) < 12.0:
                age_bins.append('oth')
            else:
                age_bins.append('P2')
    return pd.Series(age_bins, index=interview_age.index)
