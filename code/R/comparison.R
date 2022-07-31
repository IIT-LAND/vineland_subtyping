require(ggplot2)
require(reshape2)

subdom_feat <- 'rec|expr|person|domes|community|interper|play|coping'
dom_feat <- 'socialization|communication|living'

# BETWEEN-CLUSTER VINELAND SCORE COMPARISONS
# Input: dataframe with 'cluster' column (i.e., rename cluster_subdomain/domain/new_cluster as cluster) 
# and the desired feature level among 'subdomain/domain/'
clust_comparison <- function(df, level){
  if (level == 'subdomain'){
    features <- names(df)[grep(subdom_feat, names(df))]} else{
        features <- names(df)[grep(dom_feat, names(df))]}
  
  df$cluster <- as.factor(df$cluster)
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster', features)), 
                  id.vars=c('subjectkey', 'cluster'))
  df_long$cluster <- as.character(df_long$cluster)
  print(ggplot(df_long, aes(x=variable, y=value, fill=cluster)) +
          geom_boxplot() +
          facet_wrap(~variable, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('', level)))
  
  print("Check if clusters and interview age are independed")
  chk <- summary(aov(interview_age~cluster, df))
  print(chk)
  for (col in features){
    print(col)
    if (chk[[1]]$Pr[1]<0.05){
      if (length(unique(df$cluster))==2){    
        print(t.test(df[, col]~df$cluster))
      } else{
        print(summary(aov(df[, col]~df$cluster)))
        print(pairwise.t.test(df[, col], df$cluster, p.adjust.method = 'bonferroni'))}
    } else {
      print("ANCOVA with interview age covariates")
      anc <- aov(df[,col]~interview_age + cluster, df)
      print(Anova(anc, type='II'))
      print(summary(glht(anc, linfct = mcp(cluster = "Tukey"))))
    }
  }
}


# WITHIN-CLUSTER FEATURE COMPARISONS
# Input: dataframe with 'cluster' column and level (i.e., either 'subdomain', 'domain', 'new')
feat_comparison <- function(df, level){
  if (level == 'subdomain'){
    features <- names(df)[grep(subdom_feat, names(df))]} else{
        features <- names(df)[grep(dom_feat, names(df))]}
  
  df_long <- melt(subset(df, select=c('subjectkey', 'cluster', features)), 
                  id.vars=c('subjectkey', 'cluster'))
  df_long$cluster <- as.character(df_long$cluster)
  for (cl in sort(unique(df_long$cluster))){
    print(sprintf('Analyzing cluster %s', cl))
    print(pairwise.t.test(df_long$value[which(df_long$cluster==cl)], 
                          df_long$variable[which(df_long$cluster==cl)], 
                          p.adjust.method = 'bonferroni'))
  }
  print(ggplot(df_long, aes(x=cluster, y=value, fill=variable)) +
          geom_boxplot() +
          facet_wrap(~cluster, scale="free") +
          coord_cartesian(ylim = c(min(df_long$value), max(df_long$value))) +
          ggtitle(sprintf('', level)))
}


# Run the between/within post-hoc analyses
# Input: data folder name, file name with the dataframe to read, period (e.g., 'P1/2'), 
# level (i.e., 'subdomain', 'domain', 'new')
run_comparison <- function(foldername, filename_ts, period, level){
  if (level=='subdomain'){
    cluster_str <- 'cluster_subdomain'
  } else{ if (level=='domain'){
    cluster_str <- 'cluster_domain'
  } else{
    cluster_str <- 'new_cluster'
  }
  }
  df <- read.table(file.path(foldername, filename_ts),
                   sep=',',
                   header=TRUE,
                   as.is=TRUE)
  df$cluster <- as.factor(df[, cluster_str])
  clust_comparison(df, level = level)
  feat_comparison(df, level = level)
}
##########################################################################################
datafolder_name <- '/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_stratification/out_new/'

# SUBDOMAIN - P1
filename_ts <- 'imputed_data_P1_ts.csv'
run_comparison(datafolder_name, filename_ts, period='P1', level='subdomain')

# DOMAIN - P1
run_comparison(datafolder_name, filename_ts, period='P1', level='domain')

# NEW - P1
run_comparison(datafolder_name, filename_ts, period='P1', level='new')

# SUBDOMAIN - P2
filename_ts <- 'imputed_data_P2_ts.csv'
run_comparison(datafolder_name, filename_ts, period='P2', level='subdomain')

# DOMAIN - P2
run_comparison(datafolder_name, filename_ts, period='P2', level='domain')

# NEW - P2
run_comparison(datafolder_name, filename_ts, period='P1', level='new')

