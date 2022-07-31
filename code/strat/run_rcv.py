import numpy as np
import pandas as pd
from strat.create_dataset import prepare_imputation, _impute, _check_na_perc
import strat.utils as ut
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from umap import UMAP
from strat.visualization import _scatter_plot, plot_metrics, plot_miss_heat
import logging
import re
import os
import csv
from reval import best_nclust_cv


class RCV:
    """
    Class initialized with the period type (either 1 or 2, see `create_levels:_generate_age_bins` method),
    the tuple of interview periods to consider for the RCV procedure, and the proportion of observations
    to be included in the test set.

    :param ptype: possible values 1 or 2
    :type ptype: int
    :param include_age: strings of the interview periods to consider
    :type include_age: tuple
    :param trts_perc: proportion of subjects in test set
    :type trts_perc: float
    """

    def __init__(self, ptype, include_age, trts_perc):
        self.ptype = ptype
        self.include_age = include_age
        self.trts_perc = trts_perc

    def prepare_cs_dataset(self, df):
        """
        Function that takes as input a longitudinal, long-format dataset
        that has already been prepared for imputation with `create_dataset:prepare_imputation`
        function.
        It returns a dictionary with keys the interview period to consider
        and values train and test datasets stratified according to
        the count of missing features (if possible), age in years, and sex. Duplicates are dropped
        and the entries with the higher percentage of available info are retained.

        :param df: longitudinal dataset
        :type df: pandas dataframe
        :return: train dictionary divided by interview period, test dictionary
        :rtype: dict, dict
        """

        # Remove duplicates and retain the row with the
        # lowest missing info
        mask = df.reset_index().duplicated(['subjectkey', 'interview_period'], keep=False)
        dfdup = df.loc[mask.tolist()].copy()
        gui_list = np.unique(dfdup.index)
        mask_drop = []
        for idx in gui_list:
            cou = dfdup.loc[idx].countna.tolist()
            tmp = [False] * dfdup.loc[idx].shape[0]
            tmp[cou.index(min(cou))] = True
            mask_drop.extend(tmp)
        df_out = pd.concat([dfdup.loc[mask_drop], df.loc[(~mask).tolist()]])
        df_out.sort_values(['subjectkey', 'interview_period'], inplace=True)

        # Create a dictionary with interview period as keys
        dict_df = {p: df_out.loc[df_out.interview_period == p]
                   for p in self.include_age}

        # Build train/test sets, stratify by NA=1 or notNA=0
        tr_dict, ts_dict = {}, {}
        for k in dict_df.keys():
            logging.info(f'Number of subjects at {k}: {dict_df[k].shape[0]}')
            try:
                idx_tr, idx_ts = train_test_split(dict_df[k].index,
                                                  stratify=dict_df[k][['hasna', 'sex', 'ageyrs_round']],
                                                  test_size=self.trts_perc,
                                                  random_state=42)
            except ValueError:
                idx_tr, idx_ts = train_test_split(dict_df[k].index,
                                                  stratify=dict_df[k][['sex', 'ageyrs_round']],
                                                  test_size=self.trts_perc,
                                                  random_state=42)
            tr_dict[k] = dict_df[k].loc[idx_tr].sort_values(['subjectkey',
                                                             'interview_period'])
            ts_dict[k] = dict_df[k].loc[idx_ts].sort_values(['subjectkey',
                                                             'interview_period'])
            logging.info(f'Number of subjects in training set: {tr_dict[k].shape[0]}')
            logging.info(f'Number of subjects in test set: {ts_dict[k].shape[0]}')
        return tr_dict, ts_dict

    def gridsearch_cv(self, df, n_neigh, na_perc, cl_range, cv_fold, iter_cv=1, save=None):
        """
        This function can be performed to decide which percentage of missing information
        to allow, and what's the best number of neighbors to consider both for the KNNImputer
        and the KNNClassifier. It takes as input the dataset as output from the function
        `create_dataset:dataset`.

        :param df: dataframe of merged instrument versions (longitudinal entries)
        :type df: pandas dataframe
        :param n_neigh: number of neighbors for imputation and classification
        :type n_neigh: list
        :param na_perc: max percentage of missing information allowed
        :type na_perc: list
        :param cl_range: range of minimum and maximum number of clusters to look for
        :type cl_range: list
        :param cv_fold: number of cross validation loop for RCV
        :type cv_fold: list
        :param iter_cv: number of repeated CV, default 1
        :type iter_cv: int
        :param save: whether to save the performance score table, defaults None
            Name of the file required
        :type save: str
        :return: best number of n_neigh, na_perc and cv_fold, and a summary of all performances
            The best performance is the one that has highest mean acc scores in both validation
                and test and the minimum mean amplitude of CIs
        :rtype: dict, pandas dataframe
        """
        logging.disable(logging.CRITICAL)
        scaler = StandardScaler()
        subdomain_feat = [c for c in df.columns if re.search('subdom', c) and not re.search('written', c)]
        domain_feat = [c for c in df.columns if re.search('totalb', c) and not re.search('composite', c)]
        if save is not None:
            with(open(os.path.join(ut.out_folder, f'{save}.csv'), 'w')) as f:
                wr = csv.writer(f, delimiter=';', lineterminator='\n')
                wr.writerow(['cv_fold', 'na_perc_thrs', 'n_neigh', 'period', 'feat_lev',
                             'N', 'nclust', 'val_acc', 'val_ci', 'test_acc'])
        transformer = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
        scores = {}
        for k in cv_fold:
            for nap in na_perc:
                dict_tr, dict_ts = self.prepare_cs_dataset(prepare_imputation(df, nap))
                strat_agesex = {
                    p: np.array(dict_tr[p].sex + dict_tr[p].ageyrs_round.astype(str))
                    for p in self.include_age}
                strat_sex = {
                    p: np.array(dict_tr[p].sex)
                    for p in self.include_age}
                for n in n_neigh:
                    scores.setdefault('cv', list()).append(k)
                    scores.setdefault('na_perc', list()).append(nap)
                    scores.setdefault('n_neigh', list()).append(n)
                    impute = KNNImputer(n_neighbors=n)
                    dict_imp = {p: _impute(dict_tr[p], dict_ts[p], impute)
                                for p in self.include_age}
                    knn = KNeighborsClassifier(n_neighbors=n)
                    # clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')
                    clust = KMeans(random_state=42)

                    relval = best_nclust_cv.FindBestClustCV(s=knn, c=clust, nfold=k, nclust_range=cl_range,
                                                            nrand=100)
                    # Run the model
                    val_misclass = []
                    test_misclass = []
                    conf_width = []
                    for p, tup in dict_imp.items():
                        # tup[0]['interview_age'] = dict_tr[p].loc[tup[0].index]["interview_age"]
                        # tup[1]['interview_age'] = dict_ts[p].loc[tup[1].index]["interview_age"]
                        scaled_tr = scaler.fit_transform(tup[0][subdomain_feat])
                        scaled_ts = scaler.transform(tup[1][subdomain_feat])
                        X_tr = transformer.fit_transform(scaled_tr)
                        X_ts = transformer.transform(scaled_ts)
                        if min(np.unique(strat_agesex[p], return_counts=True)[1]) >= k:
                            metric, ncl = relval.best_nclust(X_tr,### changed her by vero
                                                                        iter_cv=iter_cv,
                                                                        strat_vect=strat_agesex[p])
                        else:
                            metric, ncl = relval.best_nclust(X_tr,### changed her by vero
                                                                        iter_cv=iter_cv,
                                                                        strat_vect=strat_sex[p])
                        out = relval.evaluate(X_tr, X_ts, ncl)
                        ci = (1 - (metric['val'][ncl][1][0] + metric['val'][ncl][1][1]),
                              1 - (metric['val'][ncl][1][0] - metric['val'][ncl][1][1]))
                        val_misclass.append(metric['val'][ncl][0])
                        test_misclass.append(1 - out.test_acc)
                        conf_width.append(ci[1] - ci[0])
                        if save is not None:
                            with open(os.path.join(ut.out_folder, f'{save}.csv'), 'a') as f:
                                wr = csv.writer(f, delimiter=';', lineterminator='\n')
                                wr.writerow([k, nap, n, p, 'subdomain', (X_tr.shape[0], X_ts.shape[0]),
                                             ncl, 1 - metric['val'][ncl][0],
                                             ci, out.test_acc])
                        scaled_tr = scaler.fit_transform(tup[0][domain_feat])
                        scaled_ts = scaler.transform(tup[1][domain_feat])
                        X_tr = transformer.fit_transform(scaled_tr)
                        X_ts = transformer.transform(scaled_ts)
                        if min(np.unique(strat_agesex[p], return_counts=True)[1]) >= k:
                            metric, ncl = relval.best_nclust(X_tr,### changed her by vero
                                                                        iter_cv=iter_cv,
                                                                        strat_vect=strat_agesex[p])
                        else:
                            metric, ncl = relval.best_nclust(X_tr,### changed her by vero
                                                                        iter_cv=iter_cv,
                                                                        strat_vect=strat_sex[p])
                        out = relval.evaluate(X_tr, X_ts, ncl)
                        ci = (1 - (metric['val'][ncl][1][0] + metric['val'][ncl][1][1]),
                              1 - (metric['val'][ncl][1][0] - metric['val'][ncl][1][1]))
                        val_misclass.append(metric['val'][ncl][0])
                        test_misclass.append(1 - out.test_acc)
                        conf_width.append(ci[1] - ci[0])
                        if save is not None:
                            with open(os.path.join(ut.out_folder, f'{save}.csv'), 'a') as f:
                                wr = csv.writer(f, delimiter=';', lineterminator='\n')
                                wr.writerow([k, nap, n, p, 'domain', (X_tr.shape[0], X_ts.shape[0]),
                                             ncl, 1 - metric['val'][ncl][0],
                                             ci, out.test_acc])
                    scores.setdefault('avg_val_ms', list()).append(np.mean(val_misclass))
                    scores.setdefault('avg_test_ms', list()).append(np.mean(test_misclass))
                    scores.setdefault('avg_conf_width', list()).append(np.mean(conf_width))
        scores = pd.DataFrame(scores)
        minval = scores[['avg_val_ms', 'avg_test_ms', 'avg_conf_width']].apply(sum, 1).min()
        best_param = scores.loc[scores[['avg_val_ms', 'avg_test_ms', 'avg_conf_width']].apply(sum, 1) == minval]
        logging.disable(logging.NOTSET)
        logging.info(f"Best parameters selected: {best_param[['cv', 'na_perc', 'n_neigh']].to_dict('records')[0]} "
                     f"-- Scores {best_param[['avg_val_ms', 'avg_test_ms', 'avg_conf_width']].to_dict('records')[0]}")
        return best_param[['cv', 'na_perc', 'n_neigh']].to_dict('records')[0], scores

    def run_rcv(self, df, demo_info, na_perc, n_neigh, cv_fold, cl_range, iter_cv=1, scatter=False, heatmap=False):
        """
        This function performs RCV method with fixed percentage of missing information,
        number of neighbors, number of cross validation. The range of clusters to consider still has to vary
        as desired. It is possible to flag :param scatter: and :param heatmap: to enable the
        visualization of UMAP scatterplots and the percentage fo missing information per feature for each cluster.
        The input dataset is a dataframe as returned by `strat:create_dataset:dataset` function (hence
        also longitudinal datasets). Finally, distance matrices for replication analysis are stored in the
        output folder. Also imputed datasets are saved to csv files.

        :param df: dataset obtained by merging different versions of the same instrument
        :type df: pandas dataframe
        :param demo_info: demographic information
        :type demo_info: dict
        :param na_perc: percentage of missing information
        :type na_perc: float
        :param n_neigh: number of neighbors for imputation and classification
        :type n_neigh: int
        :param cv_fold: number of cross validation iterations
        :type cv_fold: int
        :param cl_range: min/max number of clusters to consider
        :type cl_range: tuple
        :param iter_cv: number of repeated CV, default 1
        :type iter_cv: int
        :param scatter: flag for UMAP scatterplot (for training and test), defaults to False
        :type scatter: bool
        :param heatmap: flag for heatmap displaying percentage of missing information
            per feature for each cluster identified by the RCV method. Defaults to False.
        :type heatmap: bool
        :return: imputed datasets with clustering labels
        :rtype: dict
        """
        flatui = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                  "#7f7f7f", "#bcbd22", "#17becf", "#8c564b", "#a55194"]
        scaler = StandardScaler()
        # subdomain_feat = [c for c in df.columns if re.search('vscore', c) and not re.search('written', c)]
        subdomain_feat = [c for c in df.columns if re.search('subdom', c) and not re.search('written', c)]
        domain_feat = [c for c in df.columns if re.search('totalb', c) and not re.search('composite', c)]

        dict_tr, dict_ts = self.prepare_cs_dataset(prepare_imputation(df, na_perc))
        strat_agesexed = {p: np.array(dict_tr[p].sex + dict_tr[p].ageyrs_round.astype(str))
                          for p in self.include_age}
        strat_sex = {p: np.array(dict_tr[p].sex)
                     for p in self.include_age}
        transformer = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
        impute = KNNImputer(n_neighbors=n_neigh)
        dict_imp = {p: _impute(dict_tr[p], dict_ts[p], impute)
                    for p in self.include_age}
        knn = KNeighborsClassifier(n_neighbors=n_neigh)
        # clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')
        clust = KMeans(random_state=42)

        relval = best_nclust_cv.FindBestClustCV(s=knn, c=clust, nfold=cv_fold, nclust_range=cl_range,
                                                nrand=100)
        # Run the model
        for p, tup in dict_imp.items():
            dict_imp[p][1]['sex'] = [demo_info[gui].sex for gui in dict_imp[p][1].index]
            dict_imp[p][1]['phenotype'] = [demo_info[gui].phenotype.replace("'", "") for gui in dict_imp[p][1].index]
            dict_imp[p][1]['race'] = [demo_info[gui].race for gui in dict_imp[p][1].index]
            dict_imp[p][1]['collection_id'] = [dict_ts[p].loc[gui].collection_id for gui in dict_imp[p][1].index]
            dict_imp[p][1]['interview_age'] = [dict_ts[p].loc[gui].interview_age for gui in dict_imp[p][1].index]

            # tup[0]['interview_age'] = dict_tr[p].loc[tup[0].index]["interview_age"]
            # tup[1]['interview_age'] = dict_ts[p].loc[tup[1].index]["interview_age"]

            scaled_tr = scaler.fit_transform(tup[0][subdomain_feat])
            scaled_ts = scaler.transform(tup[1][subdomain_feat])
            X_tr = transformer.fit_transform(scaled_tr)
            X_ts = transformer.transform(scaled_ts)
            if min(np.unique(strat_agesexed[p], return_counts=True)[1]) >= cv_fold:
                metric,ncl = relval.best_nclust(X_tr, 
                                                iter_cv = iter_cv, 
                                                strat_vect = strat_agesexed[p])
            else:
                metric,ncl = relval.best_nclust(X_tr,
                                                iter_cv = iter_cv,
                                                strat_vect = strat_sex[p])
            out = relval.evaluate(X_tr, X_ts, ncl)
            logging.info(f"Best number of clusters: {ncl}")
            logging.info(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")
            dict_imp[p][0]['cluster_subdomain'], dict_imp[p][1][
                'cluster_subdomain'] = out.train_cllab + 1, out.test_cllab + 1
            _, subj_mis = _check_na_perc(dict_ts[p][subdomain_feat])
            dict_imp[p][1]['missing_subdomain'] = list(subj_mis.values())
            plot_metrics(metric,
                         f'UMAP preprocessed dataset, RCV performance at {p}, level subdomain')
            if scatter:
                _scatter_plot(X_tr,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][0].index, out.train_cllab + 1)],
                              flatui,
                              5, 7,
                              {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in
                               sorted(np.unique(out.train_cllab + 1))},
                              title=f'Subgroups of UMAP preprocessed Vineland TRAINING '
                                    f'dataset (period: {p} -- level: subdomain)')

                _scatter_plot(X_ts,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][1].index, out.test_cllab + 1)],
                              flatui,
                              5, 7, {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in
                                     sorted(np.unique(out.test_cllab + 1))},
                              title=f'Subgroups of UMAP preprocessed Vineland TEST '
                                    f'dataset (period: {p} -- level: subdomain)')
            if heatmap:
                dict_ts[p]['cluster'] = out.test_cllab + 1
                feat = []
                values = []
                cl_labels = np.repeat(sorted(dict_ts[p].cluster.unique().astype(str)), len(subdomain_feat))
                for lab in np.unique(sorted(out.test_cllab + 1)):
                    na_feat, _ = _check_na_perc(dict_ts[p].loc[dict_ts[p].cluster == lab][subdomain_feat])
                    feat.extend(list(na_feat.keys()))
                    values.extend(list(na_feat.values()))
                plot_miss_heat(dict_ts[p], cl_labels, feat, values, period=p, hierarchy='subdomain')

            scaled_tr = scaler.fit_transform(tup[0][domain_feat])
            scaled_ts = scaler.transform(tup[1][domain_feat])
            X_tr = transformer.fit_transform(scaled_tr)
            X_ts = transformer.transform(scaled_ts)
            if min(np.unique(strat_agesexed[p], return_counts=True)[1]) >= cv_fold:
                metric, ncl= relval.best_nclust(X_tr,
                                               iter_cv=iter_cv,
                                                strat_vect=strat_agesexed[p])
            else:
                metric, ncl = relval.best_nclust(X_tr,
                                                 iter_cv=iter_cv,
                                                 strat_vect=strat_sex[p])
            plot_metrics(metric,
                         f'UMAP preprocessed dataset, RCV performance at {p}, level domain')
            out = relval.evaluate(X_tr, X_ts, ncl)
            logging.info(f"Best number of clusters: {ncl}")
            logging.info(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")
            dict_imp[p][0]['cluster_domain'], dict_imp[p][1]['cluster_domain'] = out.train_cllab + 1, out.test_cllab + 1
            _, subj_mis = _check_na_perc(dict_ts[p][domain_feat])
            dict_imp[p][1]['missing_domain'] = list(subj_mis.values())

            if scatter:
                _scatter_plot(X_tr,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][0].index, out.train_cllab + 1)],
                              flatui,
                              5, 7,
                              {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in
                               sorted(np.unique(out.train_cllab + 1))}
                              # title=f'Subgroups of UMAP preprocessed Vineland TRAINING '
                              #       f'dataset (period: {p} -- level: domain)'
                              )

                _scatter_plot(X_ts,
                              [(gui, cl) for gui, cl in zip(dict_imp[p][1].index, out.test_cllab + 1)],
                              flatui,
                              5, 7, {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in
                                     sorted(np.unique(out.test_cllab + 1))}
                              # title=f'Subgroups of UMAP preprocessed Vineland TEST '
                              #       f'dataset (period: {p} -- level: domain)'
                              )
            if heatmap:
                dict_ts[p]['cluster'] = out.test_cllab + 1
                feat = []
                values = []
                cl_labels = np.repeat(sorted(dict_ts[p].cluster.unique().astype(str)), len(domain_feat))
                for lab in np.unique(sorted(out.test_cllab + 1)):
                    na_feat, _ = _check_na_perc(dict_ts[p].loc[dict_ts[p].cluster == lab][domain_feat])
                    feat.extend(list(na_feat.keys()))
                    values.extend(list(na_feat.values()))
                plot_miss_heat(dict_ts[p], cl_labels, feat, values, period=p, hierarchy='subdomain')

        logging.info("Saving train/test datasets with new cluster")
        new_dict_imp = relabel(dict_imp, plot_scatter=False)
        for p in new_dict_imp.keys():
            new_dict_imp[p][0].to_csv(os.path.join(ut.out_folder, f'imputed_data_{p}_tr.csv'),
                                      index_label='subjectkey')
            new_dict_imp[p][1].to_csv(os.path.join(ut.out_folder, f'imputed_data_{p}_ts.csv'),
                                      index_label='subjectkey')

        logging.info("Building distance matrix...")
        _build_distmat(imp_dict=dict_imp)

        return dict_imp


def relabel(dict_imp, plot_scatter=False):
    """
    Function that return the imputed datasets with new labels. Obtained by pasting the
    subdomain cluster labels with the domain cluster labels.

    :param dict_imp: imputed datasets
    :type dict_imp: dict
    :param plot_scatter: flag for a scatterplot with new labels
    :type plot_scatter: bool
    :return: dictionary of imputed datasets with new cluster label column
    :rtype: dict
    """
    p = list(dict_imp.keys())[0]
    transform = UMAP(random_state=42, n_neighbors=30, min_dist=0.0)
    scaler = StandardScaler()
    # subdomain_feat = [c for c in dict_imp[p][0].columns if re.search('vscore', c) and not re.search('written', c)]
    subdomain_feat = [c for c in dict_imp[p][0].columns if
                      re.search('subdom', c) and not re.search('written|cluster', c)]
    domain_feat = [c for c in dict_imp[p][0].columns if re.search('totalb', c) and not re.search('composite', c)]
    col = subdomain_feat + domain_feat
    flatui = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22", "#17becf", "#8c564b", "#a55194"]

    for p, df in dict_imp.items():
        df[0]['new_cluster'] = ['-'.join([str(a), str(b)]) for a, b in
                                zip(df[0]['cluster_subdomain'].tolist(),
                                    df[0]['cluster_domain'].tolist())]
        df[1]['new_cluster'] = ['-'.join([str(a), str(b)]) for a, b in
                                zip(df[1]['cluster_subdomain'].tolist(),
                                    df[1]['cluster_domain'].tolist())]
        scaled_tr = scaler.fit_transform(df[0][col])
        scaled_ts = scaler.transform(df[1][col])
        X_tr = transform.fit_transform(scaled_tr)
        X_ts = transform.transform(scaled_ts)

        df[0]['umap_dim1'], df[0]['umap_dim2'] = X_tr[:, 0], X_tr[:, 1]
        df[1]['umap_dim1'], df[1]['umap_dim2'] = X_ts[:, 0], X_ts[:, 1]

        if plot_scatter:
            _scatter_plot(X_tr,
                          [(gui, cl) for gui, cl in zip(df[0].index, df[0].new_cluster)],
                          flatui,
                          5, 7,
                          {str(ncl): ' '.join(['cluster', str(ncl)]) for ncl in sorted(np.unique(df[0].new_cluster))},
                          title=f'New labels Vineland {p} training set')
            _scatter_plot(X_ts,
                          [(gui, cl) for gui, cl in zip(df[1].index, df[1].new_cluster)],
                          flatui,
                          5, 7,
                          {str(ncl): ' '.join(['cluster', str(ncl)]) for ncl in sorted(np.unique(df[1].new_cluster))},
                          title=f'New labels Vineland {p} test set')
    return dict_imp


"""
Private functions
"""


def _build_distmat(imp_dict):
    """
    Function that stores distance matrices for both training and test for replication analysis.

    :param imp_dict: imputed datasets divided by interview period, as built by `RCV:run_rcv`
    """
    feat_dict = {
        # 'subdomain': [c for c in imp_dict['P1'][0].columns if re.search('vscore', c) and not re.search('written', c)],
        'subdomain': [c for c in imp_dict['P1'][0].columns if re.search('subdom', c) and not re.search('written', c)],
        'domain': [c for c in imp_dict['P1'][0].columns if re.search('totalb', c) and not re.search('composite', c)]}

    # Save distance matrices
    scaler = MinMaxScaler()
    for p, tup in imp_dict.items():
        train = tup[0]
        test = tup[1]
        for kfeat, colnames in feat_dict.items():
            df_tr = train[colnames].copy()
            df_cl_tr = pd.DataFrame({'subjectkey': df_tr.index, 'cluster': imp_dict[p][0][f'cluster_{kfeat}']},
                                    index=df_tr.index).sort_values('cluster')
            sc_tr = scaler.fit_transform(df_tr.loc[df_cl_tr.index])
            distmat_tr = pd.DataFrame(cdist(sc_tr, sc_tr), columns=df_cl_tr.index)
            distmat_tr.index = df_cl_tr.index
            distmat_tr[f'cluster_{kfeat}'] = df_cl_tr['cluster'].astype(str)

            df_ts = test[colnames].copy()
            df_cl_ts = pd.DataFrame({'subjectkey': df_ts.index, 'cluster': imp_dict[p][1][f'cluster_{kfeat}']},
                                    index=df_ts.index).sort_values('cluster')
            sc_ts = scaler.fit_transform(df_ts.loc[df_cl_ts.index])
            distmat_ts = pd.DataFrame(cdist(sc_ts, sc_ts), columns=df_cl_ts.index)
            distmat_ts.index = df_cl_ts.index
            distmat_ts[f'cluster_{kfeat}'] = df_cl_ts['cluster'].astype(str)

            with open(os.path.join(ut.out_folder, f'vineland_distmat{kfeat}TR{p}.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerow(['subjectkey'] + distmat_tr.columns.tolist())
                for gui, row in distmat_tr.iterrows():
                    wr.writerow([gui] + row.tolist())
            with open(os.path.join(ut.out_folder, f'vineland_distmat{kfeat}TS{p}.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerow(['subjectkey'] + distmat_ts.columns.tolist())
                for gui, row in distmat_ts.iterrows():
                    wr.writerow([gui] + row.tolist())
