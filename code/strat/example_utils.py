"""
Fill in the variables as requested and rename the file as
utils.py
"""

# Folder to txt NDAR tables
data_folder = ''

# Name of the file with phenotypes for subject selection
# Include file extension
phenotype_file = ''

# Output folder
out_folder = ''

# Number of neighbors to consider for KNN imputation
neighbors = 5

# Add table file names (file extension included). In the example, vineland-survey,
# vineland-parent, and vineland-III are requested.
instrument_dict = {
    'vinelands': '',
    'vinelandp': '',
    'vineland3': ''}

# Insert column names shared by all the instruments
shared_col = ['collection_id',
              'interview_age',
              'interview_date',
              'sex']

# Vineland column names to include from all version datasets
# Subdomain range [1, 24]
# Domain range [20, 140]
vineland_col_names = ['relationship',
                      'collection_id',
                      'receptive_vscore',
                      'receptivesubdomain_1a',
                      'expressive_vscore',
                      'expressivesubdomain_2a',
                      'written_vscore',
                      'writtensubdomain_3a',
                      'personal_vscore',
                      'personalsubdomain_1a',
                      'domestic_vscore',
                      'domesticsubdomain_2a',
                      'community_vscore',
                      'communitysubdomain_3a',
                      'interprltn_vscore',
                      'interpersrelationsubdom_1a',
                      'playleis_vscore',
                      'playleisuretimesubdomain_2a',
                      'copingskill_vscore',
                      'copingskillssubdomain_3a',
                      'communicationdomain_totalb',
                      'livingskillsdomain_totalb',
                      'socializationdomain_totalb',
                      'composite_totalb']
# Renaming patters, keys: new names, values: names of the columns to replace
vineland_mergecol = {'receptive_vscore': 'receptivesubdomain_1a',
                     'expressive_vscore': 'expressivesubdomain_2a',
                     'written_vscore': 'writtensubdomain_3a',
                     'personal_vscore': 'personalsubdomain_1a',
                     'domestic_vscore': 'domesticsubdomain_2a',
                     'community_vscore': 'communitysubdomain_3a',
                     'interprltn_vscore': 'interpersrelationsubdom_1a',
                     'playleis_vscore': 'playleisuretimesubdomain_2a',
                     'copingskill_vscore': 'copingskillssubdomain_3a'}

# DO NOT MODIFY below
# Regular expression to filter subjects with ASC diagnosis from NDAR dataset
phenotype_regex = r'^(?!no|sub|[adhd, ]{0,6}p[r]{0,1}[eo][sv]|at risk for|broad[er]{0,2}|' \
                  r'd[d]{0,1}[elays]{0,5}|[rule-]{0,5}o[u]{0,1}t[her]{0,3}|td|hx|typsib).' \
                  r'*(pdd|\bau[s]{0,1}t[ictsm]{1,5}\b(?!\?)|asd(?!\?| \(by hx\)|.sib|.sx)|\_' \
                  r'\b)|autismspectrumdisorder|299|\baut\b(?!feat)|(non control)|asp' \
                  r'(?!ects)|asperger|ados|^[lh]fa|[hl]fa\-[hl]fa'
# for these terms check for asd|autism spectrum and asd in
# phenotype_description column, respectively
exception_regex = 'non control|calculated'
