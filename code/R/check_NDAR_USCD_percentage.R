library(readxl)
ACE_Lab_NDAR_Collections <- read_excel("Desktop/ACE_Lab_NDAR_Collections.xlsx")

ACE_Lab_NDAR_Collections <- read_excel("Desktop/ACE_Lab_NDAR_Collections.xlsx")

collection_vabs_NDAR <- read_excel("OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/results/FINAL_correct/collection_vabs_NDAR.csv")
unique(collection_ids$collection_id)

###########################################################################################################################
# "3127" --> original NDAR NO --> our subset of NDAR NO
# "0009" --> original NDAR YES --> our subset of NDAR NO --> we manually exclude those subject before running reval
# "2115" --> original NDAR YES --> our subset of NDAR YES --> 0.18 
# "2290" --> original NDAR YES --> our subset of NDAR NO --> Subject older than 6yo

###### the problem is 2115 ######
#chek if the subject ID from NDAR are also in the longitudinal
ucsd_long_allSubj_clusters <- read.csv("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/FINAL_correct/ucsd_long_allSubj_clusters.csv", header=TRUE)
NDAR_ts <- read.csv("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/FINAL_correct/imputed_data_P1_ts.csv")
NDAR_tr <- read.csv("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/FINAL_correct/imputed_data_P1_tr.csv")

col= c('subjectkey', 'communicationdomain_totalb', 'livingskillsdomain_totalb', 'socializationdomain_totalb', 'motorskillsdomain_totalb', 'sex','phenotype',  'race' ,'collection_id', 'interview_age', 'cluster_domain')
NDAR_2115 = rbind(NDAR_ts[NDAR_ts$collection_id == "2115",col],NDAR_tr[NDAR_tr$collection_id == "2115",col])
############
# 2 subjects are in those collection ID that were included in REVAL
## NDARJM528WDQ , NDARVT481KVL


#### original UCSD data 
original_ucsd = read.csv("/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/data/raw/ucsd/LWReport_04182020_for_ML.csv")
# no way of find NDAR ids here

### original NDAR
vinelandparent_200503 <- read.delim("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/data/raw/ndar/vinelandparent_200503.txt", header=TRUE)
vinland301 <- read.delim("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/data/raw/ndar/vinland301.txt", header=TRUE)
vinelandsurvey_200505 <- read.delim("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat/data/raw/ndar/vinelandsurvey_200505.txt", header=TRUE)
subject_2115_NDAR = vinelandsurvey_200505[vinelandsurvey_200505$subjectkey %in% c('NDARJM528WDQ','NDARVT481KVL'),]

NDAR_sub_ID = unique(subject_2115_NDAR$src_subject_id)

NDAR_UCSD_2115_sub = ucsd_long_allSubj_clusters[ucsd_long_allSubj_clusters$subjectid %in% NDAR_sub_ID,]

##################################### Karen's requested analysis on % of subject we can find also on NDAR #########################################################
# new download of UCSD from NDAR
VINE_1 <- read.delim("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data/raw/ucsd/vinelandparent_200502_lastrelease_july22.txt", header=TRUE)
VINE_2 <- read.delim("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data/raw/ucsd/vinelandparent_200503_lastreliese_july22.txt", header=TRUE)
VINE_3 <- read.delim("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data/raw/ucsd/vinelandsurvey_200505_lastreliese_july22.txt", header=TRUE)

VINE_1 = VINE_1[-1,]
VINE_2 = VINE_2[-1,]
VINE_3 = VINE_3[-1,]

# get subject list
sublist_lastrelease = c(VINE_1$src_subject_id,VINE_2$src_subject_id,VINE_3$src_subject_id)
print(length(unique(sublist_lastrelease)))

sub_ids_lasrelise_NDA = as.data.frame(sublist_lastrelease)
results_path = "~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/"
write.csv(sub_ids_lasrelise_NDA,file.path(results_path,'sub_ids_lasrelise_NDA.csv'))

## original USCD database given by Karen that were used for the anlysis, after the preprocessed(<72 months at T1, with no missing values etc)
original_uscd<- read.csv("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/FINAL_correct/ucsd_long_allSubj_clusters.csv", header=TRUE)
# subsetting only ASD and TD
original_uscd = subset(original_uscd,original_uscd$cluster %in% c("TD","1.0","3.0","2.0"))
# get sublist
sublist_original = unique(original_uscd$subjectid)
print(length(sublist_original))

sub_ids_original = as.data.frame(sublist_original)
write.csv(sub_ids_original,file.path(results_path,'sub_ids_we_used.csv'))


#0.2724409
#check how many ASD 
original_uscd_ASD =subset(original_uscd,original_uscd$cluster %in% c("1.0","3.0","2.0"))
sublist_original_ASD = unique(original_uscd_ASD$subjectid)
length(sublist_original_ASD)
sub_ids_origina_asd = as.data.frame(sublist_original_ASD)
write.csv(sub_ids_origina_asd,file.path(results_path,'sub_ids_we_used_asd_only.csv'))

#0.2516447

# compare subject key from last release from those given directly by karen
#compute the % - > intectect / total subject used
print('percentage of subjects shared between Karen s database and NDAR')
print((length(intersect(sublist_lastrelease,sublist_original)))/(length(sublist_original)))
commun_subjects = intersect(sublist_lastrelease,sublist_original)

# subject used not in NDAR
uncommon_subjects_Karensonly = sublist_original[!sublist_original %in% sublist_lastrelease]

# subject in NDAR we did not used (probably older)
uncommon_subjects_NDARonly = sublist_lastrelease[!sublist_lastrelease %in% sublist_original]

## only ASD
print('percentage of ASD subjects shared between Karen s database and NDAR')
print((length(intersect(sublist_lastrelease,sublist_original_ASD)))/(length(sublist_original_ASD)))
commun_subjects_asd = intersect(sublist_lastrelease,sublist_original_ASD)

############################################################################################################################
# DATA BEFORE PREPROCESSING

# databse given by karen before the preprocessing
original_ucsd_raw = read.csv("/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/data/raw/ucsd/LWReport_04182020_for_ML.csv")

# get sub_list
sublist_original_raw = unique(original_ucsd_raw$subjectid)
print(length(sublist_original_raw))

# compare subject key from last release from those given directly by karen before preprocessing
# compute the % - > intectect / total subject used
print('percentage of subjects shared between Karen s database BEFORE PREPROCESSING and NDAR')
print((length(intersect(sublist_lastrelease,sublist_original_raw)))/(length(sublist_original_raw)))




#################################################################################################################
# col2use = c("collection_id", "src_subject_id", "interview_age", "subjectkey", "communicationdomain_totalb", "livingskillsdomain_totalb" ,"socializationdomain_totalb","motorskillsdomain_totalb")
# 
# # check 
# VINE = rbind(VINE_1[,col2use],VINE_2[,col2use],VINE_3[,col2use])
# VINE_only = VINE[VINE$src_subject_id %in% uncommon_subjects_NDARonly,]
# 
# original_only = original_uscd[original_uscd$subjectid %in% uncommon_subjects_Karensonly,]
# 
# 
# table(as.numeric(original_only$interview_age))
# 


