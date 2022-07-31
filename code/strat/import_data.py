from collections import namedtuple
import pandas as pd
import pickle as pkl
import re
import os
import logging
import strat.utils as ut

# Namedtuple for storing demographic information
# (i.e., sex, phenotype, race)
DemoInfo = namedtuple('DemoInfo', ['sex', 'phenotype', 'race'])


class ReadData:
    """
    The ReadData class enables the data_wrangling method and the assess_longitudinal method.
    It is initialized by
    :param instrument_file_names: a dictionary with the descriptive names of the instrument we want to import as keys,
        and the corresponding name of the data files as values.
    :type instrument_file_names: dict
    :param pheno_file_name: name of the file with subject phenotypes
    :type pheno_file_name: str
    :param phenotype: string specifying the phenotype of the subjects we want to consider. Two options
        have been implemented so far: (1) "autism" for the individuals who received an ASD
        diagnosis and (2) None to import all the subjects available in NDAR.
    :type phenotype: str
    """

    def __init__(self, instrument_file_names,
                 pheno_file_name,
                 phenotype=None):
        self.instrument_file_name = instrument_file_names
        self.pheno_file_name = pheno_file_name
        self.phenotype = phenotype  # only 'autism' enabled for now

    def data_wrangling(self, save=True):
        """
        This method (1) selects the subjects with a specific (if required) diagnosis. In order to
        do this the user needs to create a utils.py file with a variable "data_folder", that stores the
        path to the data, and a variable "phenotype_file", that stores the name of the table to read
        in order to select the subjects who received the diagnosis required. (2) Creates and dumps a dictionary of
        namedtuples objects that store the demographic information of the selected individuals.
        (3) Reads the dataset of the instrument desired and stores them in a dictionary ordered wrt subjectkey
        and interview age, with the
        descriptive names of the instruments as keys and the dataframes with the observation of the
        subjects selected by phenotype as values.

        In order to input the regular expression to select all the possible strings describing the
        phenotype of interest, the utils.py file should include two variables (for the autistic profile).
        Namely, the phenotype_regex and the exception_regex (see _select_phenotype private function below).

        :param save: save demographic information, defaults True
        :type save: bool
        :return: dictionary as created at step 3, demographic info
        :rtype: dict, dictionary of namedtuples
        """
        # Select individuals with desired phenotype
        if self.phenotype == 'autism':
            logging.info(f"Select individuals with {self.phenotype} and save "
                         f"demographic info.")
            gui_index, demo_info = _select_phenotype(os.path.join(ut.raw_ndar_data_folder, ##vero changed the path here/utils
                                                                  self.pheno_file_name),
                                                     ut.phenotype_regex,
                                                     ut.exception_regex)
            logging.info(f'Number of subjects with {self.phenotype}: {len(demo_info)}\n\n')
        else:
            logging.info('Phenotype not selected, considering all subjects in the database.\n\n')
            gui_index, demo_info = _select_phenotype(os.path.join(ut.raw_ndar_data_folder, ##vero changed the path here/utils
                                                                  self.pheno_file_name))
        if save:
            # Dumping the demographic info
            pkl.dump(demo_info, open(os.path.join(ut.out_folder, 'demographics.pkl'), 'wb')) ## here is out folder
        # Read tables
        logging.info('Loading datasets:')
        table_dict = {}
        for tab_name in self.instrument_file_name.values():
            table_dict[tab_name.replace('.txt', '')] = _read_ndar_table(tab_name,
                                                                        gui_index,
                                                                        ut.raw_ndar_data_folder)
        return table_dict, demo_info


"""
Private functions
"""


def _read_ndar_table(tab_name, gui_index, data_folder):
    """
    Private function used by data_wrangling method to import instrument datasets.

    :param tab_name: name of the dataset to read
    :type tab_name: str
    :param gui_index: GUIs of subjects to consider (e.g., specific phenotype)
    :type gui_index: pandas Index
    :param data_folder: name of the folder where data are stored
    :type data_folder: str
    :return: imported data, sorted wrt subjectkey and interview_age
    :rtype: pandas dataframe
    """
    name = tab_name.replace(".txt", '')
    logging.info(f'{name}')
    ins_table = pd.read_csv(os.path.join(ut.raw_ndar_data_folder, tab_name),### changed here by vero
                            sep='\t', header=0, skiprows=[1],
                            low_memory=False,
                            parse_dates=['interview_date'])
    # Ordered by interview age and subjectkey
    ins_table.sort_values(by=['subjectkey',
                              'interview_age'],
                          axis=0, inplace=True)
    ins_table.index = ins_table['subjectkey']
    ins_table.drop('subjectkey', axis=1, inplace=True)
    ins_table = ins_table.loc[gui_index.intersection(ins_table.index).unique()]
    logging.info(f'Read table {name} -- N subjects (unique): {ins_table.shape[0]} '
                 f'({ins_table.index.unique().shape[0]})\n')
    return ins_table


def _select_phenotype(file_path,
                      phenotype_regex=None,
                      exception_regex=None):
    """
    Function that reads the NDAR phenotype file, filter the desired phenotypes
    (designed for autistic phenotype) and returns the pandas Index Series of unique GUIs and
    the dictionary of namedtuple objects storing demographic info.

    :param file_path: Path to the file and file name where phenotypic information are stored.
    :type file_path: str
    :param phenotype_regex: Regular expression to detect the autistic phenotype,
        defaults to None (all subjects are considered).
    :type phenotype_regex: str
    :param exception_regex: Regular expression exceptions for autistic phenotypes (i.e., 'non control|calculated').
        For these cases the phenotype_description column should match the phenotype_regex regular expression.
        Otherwise the subject is dropped. Defaults to None (all subjects are considered).
    :type exception_regex: str
    :return: unique subject GUI with desired phenotype, demographic information of the
        selected subjects
    :rtype: pandas Index Series, dictionary of namedtuple
    """
    phenotype_table = pd.read_table(file_path,
                                    sep='\t',
                                    header=0,
                                    skiprows=[1],
                                    low_memory=False,
                                    index_col='subjectkey')
    if phenotype_regex is not None and exception_regex is not None:
        # Drop rows with missing phenotype
        phenotype_table.dropna(subset=['phenotype'], inplace=True)

        # Save GUI subjects with required phenotype
        subjindex = phenotype_table[list(map(lambda x: bool(re.search(phenotype_regex,
                                                                      str(x).lower())),
                                             phenotype_table.phenotype))].index
        phenotype_table = phenotype_table.loc[subjindex]
        # Check if all duplicate entries are consistent,
        # (i.e., they all have accepted strings).
        # Otherwise, add to list of subjects to drop.
        subjout = set()
        dupsubj = phenotype_table.loc[phenotype_table.index.duplicated(keep=False)].index.unique()
        for idsubj in dupsubj:
            uniquediag = set([str(d).lower() for d in phenotype_table.loc[idsubj].phenotype.unique()])
            if len(uniquediag) > 1:
                for d in uniquediag:
                    ctr = bool(re.search(phenotype_regex, d))
                    if not ctr:
                        subjout.add(idsubj)
        phenotype_table.drop(subjout, axis=0, inplace=True)
        # Save exception phenotype
        checksubj = phenotype_table[list(map(lambda x: bool(re.search(exception_regex,
                                                                      str(x).lower())),
                                             phenotype_table.phenotype))].index
        dropexc = set()
        for s in checksubj:
            if not re.search(phenotype_regex,
                             str(phenotype_table.loc[s].phenotype_description).lower()):
                dropexc.add(s)
        phenotype_table.drop(dropexc, axis=0, inplace=True)
    else:
        subjindex = phenotype_table.index.unique()
        phenotype_table = phenotype_table.loc[subjindex]

    # Drop duplicates with all allowed phenotypes
    phenotype_table = phenotype_table.loc[~phenotype_table.index.duplicated(keep='first')]

    demodict = {gui: DemoInfo(sex=row.sex,
                              phenotype=str(row.phenotype).lower(),
                              race=row.race.lower())
                for gui, row in phenotype_table.iterrows()}

    return phenotype_table.index, demodict
