require(ggplot2)
require(ggpubr)

# CONFOUNDERS (functions)

# Percentage of missing information (ANOVA with pairwise t-test)
# Input: dataframe with columns cluster (factor) and missing


##########################################################################################
work_dir = '/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat'
data_path = file.path(work_dir,'results','vero_results')
result_path = file.path(work_dir,'results','vero_results')
plot_path = file.path(work_dir,'plots')
##########################################################################################

miss_info <- function(df){
  if (length(unique(df$cluster)) == 2){
    print(t.test(df$missing~df$cluster))
  } else{
    print(summary(aov(df$missing~df$cluster)))
    print(pairwise.t.test(df$missing, df$cluster))}
}
datafolder_name
# Sex
# Input: dataframe with sex and cluster columns
sex <- function(df){
  tabsex_sub <- table(df$sex, df$cluster)
  print(tabsex_sub)
  print(chisq.test(tabsex_sub))
}

# Collection id. Only test labels that appear more than once.
# Input: dataframe with cluster and collection_id columns
collection <- function(df){
  ridcollection <- tapply(df$cluster, df$collection_id, function(x) {length(unique(x))})
  newdf <- df[df$collection_id %in% sort(unique(df$collection_id))[ridcollection>(length(unique(df$cluster))-1)],]
  print(table(newdf$collection_id, newdf$cluster))
  if (dim(newdf)[1]>0){
  print(chisq.test(table(newdf$collection_id, newdf$cluster)))} else{
    print('Chi-squared test not possible')
  }
}

# Phenotype. Test only phenotypes that appear more than once.
# Input: dataframe wth cluster and phenotype columns
phenotype <- function(df){
  ridtabpheno <- tapply(df$cluster, df$phenotype, function(x) {length(unique(x))})
  newdf <- df[df$phenotype %in% sort(unique(df$phenotype))[ridtabpheno>(length(unique(df$cluster))-1)],]
  print(table(newdf$phenotype, newdf$cluster))
  print(chisq.test(table(newdf$phenotype, newdf$cluster)))
}

# Interview age. Compare interview ages and plot boxplots with p-value comparisons.
# Input: dataframe with cluster and interview_age as columns
interview_age <- function(df){
  if (length(unique(df$cluster))==2){
    print(t.test(df$interview_age~df$cluster))
    mycomp <- list(as.character(sort(unique(df$cluster))))
  } else{
    print(summary(aov(interview_age~cluster, df)))
    print(pairwise.t.test(df$interview_age, df$cluster))
    mycomp <- list()
    i <- 1
    for (cl in 1:(length(sort(unique(df$cluster)))-1)){
      for (j in (cl+1):length(sort(unique(df$cluster)))){
        mycomp[[i]] <- c(as.character(as.character(sort(unique(df$cluster)))[cl]), 
                         as.character(as.character(sort(unique(df$cluster)))[j]))
        i <- i+1}}
    }
  print(ggplot(df, aes(x=as.factor(cluster), y=interview_age)) + 
        geom_boxplot() + 
        ggtitle(print('')) +
        xlab("Clusters") +
        ylab("Interview age") +
        geom_jitter(shape=16, position=position_jitter(0.2)) +
        stat_summary(fun=mean, geom="point", shape=21, size=4) + 
        stat_compare_means(comparisons = mycomp, method = "t.test", 
                           symnum.args = list(cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1), 
                                              symbols = c("****", "***", "**", "*", "ns"))))
}

# RUN CODE (function)
#Input: name of the data folder, name of the test set file, period (e.g., 'P1'), 
#and level (i.e., 'subdomain'/'domain/new').
run_confounders <- function(foldername, filename_ts, period, level){
  if (level=='subdomain'){
    cluster_str <- 'cluster_subdomain'
    missing_str <- 'missing_subdomain'
  } else{ if (level=='domain'){
    cluster_str <- 'cluster_domain'
    missing_str <- 'missing_domain'
  } else{
    cluster_str <- 'new_cluster'
  }
  }
  df <- read.table(file.path(foldername, filename_ts),
                      sep=',',
                      header=TRUE,
                      as.is=TRUE)
  df$cluster <- as.factor(df[, cluster_str])
  if (level=='new'){
    df$missing <- df$missing_subdomain + df$missing_domain
  } else{
    df$missing <- df[, missing_str]
  }
  miss_info(df)
  sex(df)
  collection(df)
  phenotype(df)
  interview_age(df)
}

##########################################################################################
datafolder_name <- data_path

# SUBDOMAIN - P1
filename_ts <- 'imputed_data_P1_ts.csv'
run_confounders(datafolder_name, filename_ts, period='P1', level='subdomain')

# DOMAIN - P1
run_confounders(datafolder_name, filename_ts, period='P1', level='domain')

# NEW - P1
run_confounders(datafolder_name, filename_ts, period='P1', level='new')

# SUBDOMAIN - P2
filename_ts <- 'imputed_data_P2_ts.csv'

run_confounders(datafolder_name, filename_ts, period='P2', level='subdomain')

# DOMAIN - P2
run_confounders(datafolder_name, filename_ts, period='P2', level='domain')

# NEW - P2
run_confounders(datafolder_name, filename_ts, period='P1', level='new')

