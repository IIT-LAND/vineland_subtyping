"""
In the following we build a longitudinal dataset (long format) from test set at Pn and we return a csv file
where for each subject the new cluster label is reported. This will then be used to investigate trajectories
for the different groups.
"""

from strat.create_dataset import _impute, prepare_imputation
import pandas as pd
import logging


def build_long(long_df, test_df, perc_na, method, period='P1'):
    logger = logging.getLogger()

    logging.info(f'Creating the longitudinal dataset (long format) from test set at period {period}...')
    # Because subject with interview age < 12 months lack several subdomains, we drop them.
    # Drop subjects with only one observation or with percentage of NAs > perc_na
    # and save the guis of those that we consider (intersected with test set guis)
    start_dim = long_df.shape[0]
    logger.disabled = True
    long_df = prepare_imputation(long_df.loc[long_df.interview_age > 12], perc_na)
    logger.disabled = False
    gui_mt = long_df.reset_index().duplicated('subjectkey', keep=False)
    gui_mt.index = long_df.index

    long_df = long_df.loc[gui_mt]
    gui_list = long_df.index.unique().intersection(test_df.index)

    logging.info(f'Initial number of subjects in longitudinal dataset: N={start_dim}')
    logging.info(f'Dropped N={start_dim - long_df.shape[0]} subjects.')
    logging.info(f'Current number of subjects: {long_df.shape[0]}')

    # New longitudinal dataset with repeated measures in addition to those already considered at P1
    long_new = pd.DataFrame()
    drop_gui = []
    # Append repeated measures (different than the ones already included in test set)
    for gui in gui_list:
        df_gui = long_df.loc[gui]
        if df_gui.loc[df_gui.interview_age != test_df.loc[gui].interview_age].shape[0] > 0:
            long_new = long_new.append(df_gui.loc[df_gui.interview_age != test_df.loc[gui].interview_age])
        else:
            drop_gui.append(gui)

    # Drop single observations
    gui_list = gui_list.drop(drop_gui)

    # Impute elements !=P1, > P1 from longitudinal dataset
    equal_bool = long_new.loc[long_new.interview_age <= 72]
    equal_bool = equal_bool.loc[equal_bool.interview_age >= 12]
    greater_bool = long_new.loc[long_new.interview_age <= 156]
    greater_bool = greater_bool.loc[greater_bool.interview_age > 72]
    adult_bool = long_new.loc[long_new.interview_age > 156]

    equal_df = _impute(equal_bool,
                       equal_bool, method)[0]
    equal_df = pd.concat([equal_df,
                          equal_bool[[c for c in long_new.columns if c not in equal_df.columns]],
                          test_df.loc[equal_df.index][[c for c in test_df.columns if c not in equal_bool.columns]]],
                         axis=1)

    greater_df = _impute(greater_bool,
                         greater_bool, method)[0]
    greater_df = pd.concat([greater_df,
                            greater_bool[[c for c in long_new.columns if c not in greater_df.columns]],
                            test_df.loc[greater_df.index][
                                [c for c in test_df.columns if c not in greater_bool.columns]]],
                           axis=1)

    try:
        adult_df = _impute(adult_bool,
                           adult_bool, method)[0]
        adult_df = pd.concat([adult_df,
                              adult_bool[[c for c in long_new.columns if c not in adult_df.columns]],
                              test_df.loc[adult_df.index][[c for c in test_df.columns if c not in adult_bool.columns]]],
                             axis=1)

        long_imp = pd.concat([test_df.loc[gui_list], equal_df[[c for c in test_df.columns]],
                              greater_df[[c for c in test_df.columns]],
                              adult_df[[c for c in test_df.columns]]])
    except ValueError:
        long_imp = pd.concat([test_df.loc[gui_list], equal_df[[c for c in test_df.columns]],
                              greater_df[[c for c in test_df.columns]]])

    long_imp.reset_index(inplace=True)
    long_imp.sort_values(['subjectkey', 'interview_age'], inplace=True)
    long_imp.index = long_imp.subjectkey
    long_imp.drop('subjectkey', inplace=True, axis=1)

    time = []
    for gui in long_imp.index.unique():
        time.extend([i + 1 for i in range(long_imp.loc[gui].shape[0])])
    long_imp.insert(0, 'time', time)
    logging.info(f'Number of subjects with at least two entries at {period}: N={long_imp.index.unique().shape[0]}')
    logging.info("Saving dataset to csv file...")

    # Save to csv file
    long_imp.to_csv(f'./out/long{period.lower()}_imputed.csv')

    return long_imp
