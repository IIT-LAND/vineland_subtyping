import os

"""
This file includes paths to folders, file names, parameters,
and other information useful for the data pre-processing step.
"""

# Folder to txt NDAR tables
raw_data_folder = os.path.join('~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data/raw')
raw_ndar_data_folder = os.path.join('~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data/raw/ndar')
tidy_data_folder = os.path.join(os.path.expanduser('~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data'), 'tidy')

# Name of the file with phenotypes for subject selection
phenotype_file = 'ndar_phenotypes.txt'

# change if you are working with old subj.
# Output folder - young
#out_folder = os.path.join(os.path.expanduser('~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/results/'), 'vero_results')

# Output folder - old
#out_folder = os.path.join(os.path.expanduser('~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/results/'), 'old_subj_results')

out_folder = os.path.join(os.path.expanduser('~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/'), 'FINAL_correct')

# Number of neighbors for imputation technique
neighbors = 5

# Dictionary with names of Vineland files as input
# to the dataset() function
instrument_dict = {
    'vinelands': 'vinelandsurvey_200505.txt',
    'vinelandp': 'vinelandparent_200503.txt',
    'vineland3': 'vinland301.txt'
}

# Column names shared by all the instruments
fixed_col = ['collection_id',
             'interview_age',
             'interview_date',
             'edition']
# Fix the column names for subdomains/domains, raw scores
vineland_col_names = ['relationship',
                      'receptivesubdomain_1',
                      'expressivesubdomain_2',
                      'writtensubdomain_3',
                      'personalsubdomain_1',
                      'domesticsubdomain_2',
                      'communitysubdomain_3',
                      'interpersrelationsubdom_1',
                      'playleisuretimesubdomain_2',
                      'copingskillssubdomain_3',
                      'communicationdomain_totalb',
                      'livingskillsdomain_totalb',
                      'socializationdomain_totalb',
                      'motorskillsdomain_totalb', ### add here
                      'composite_totalb']
# Vineland rename column patterns. Keys are the new column names.
vineland_mergecol = {
    # 'receptive_vscore': 'receptivesubdomain_1a',
    #                  'expressive_vscore': 'expressivesubdomain_2a',
    #                  'written_vscore': 'writtensubdomain_3a',
    #                  'personal_vscore': 'personalsubdomain_1a',
    #                  'domestic_vscore': 'domesticsubdomain_2a',
    #                  'community_vscore': 'communitysubdomain_3a',
    #                  'interprltn_vscore': 'interpersrelationsubdom_1a',
    #                  'playleis_vscore': 'playleisuretimesubdomain_2a',
    #                  'copingskill_vscore': 'copingskillssubdomain_3a',
    'relationship': 'respondent',
    'livingskillsdomain_totalb': 'dailylivsk_stnd_score'}

# subdomain thresholds, maximum between Vineland 2 and 3 is selected
vinesubdom_thrs = {'receptivesubdomain_1': 78,
                   'expressivesubdomain_2': 108,
                   'writtensubdomain_3': 76,
                   'personalsubdomain_1': 110,
                   'domesticsubdomain_2': 60,
                   'communitysubdomain_3': 116,
                   'interpersrelationsubdom_1': 86,
                   'playleisuretimesubdomain_2': 72,
                   'copingskillssubdomain_3': 66}

ucsd_colnames = r'vine_agemo_*|vine_ComExprv_Raw_*|vine_DlyPers_Raw_*|vine_DlyDmstc_Raw_*|' \
                r'vine_DlyCmty_Raw_*|vine_SocIntPers_Raw_*|vine_SocLeisr_Raw_*|' \
                r'vine_SocCope_Raw_*|vine_ComTotal_DomStd_*|vine_DlyTotal_DomStd_*|' \
                r'vine_SocTotal_DomStd_*'

# SELECT ASD PHENOTYPES
# Regular expression to filter subjects with ASC diagnosis from NDAR dataset
phenotype_regex = r'^(?!no|sub|[adhd, ]{0,6}p[r]{0,1}[eo][sv]|at risk for|broad[er]{0,2}|' \
                  r'd[d]{0,1}[elays]{0,5}|[rule-]{0,5}o[u]{0,1}t[her]{0,3}|td|hx|typsib).' \
                  r'*(pdd|\bau[s]{0,1}t[ictsm]{1,5}\b(?!\?)|asd(?!\?| \(by hx\)|.sib|.sx)|\_' \
                  r'\b)|autismspectrumdisorder|299|\baut\b(?!feat)|(non control)|asp' \
                  r'(?!ects)|asperger|ados|^[lh]fa|[hl]fa\-[hl]fa|(calculated)'
# For the following terms check for asd|autism spectrum and asd in
# phenotype_description column of the ndar_subject01.txt file, respectively
exception_regex = 'non control|calculated'
