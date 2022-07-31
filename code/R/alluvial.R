require(ggalluvial)
require(ggplots)

# # ALLUVIAL PLOTS
# # Plot alluvials for a fixed interview period. Highlight sex of the subjects.
# # Input: dataframe with cluster_subdomain, cluster_domain, and sex columns; plot title
# ##########################################################################################
# work_dir = '/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat'
# data_path = file.path(work_dir,'results','vero_results')
# result_path = file.path(work_dir,'results','vero_results')
# plot_path = file.path(work_dir,'plots')
##########################################################################################

plotalluvial <- function(df,subjectkey,sex,cluster1,cluster2, plot_title){
  df <- df[order(df[,subjectkey]),]
  name_col <- paste('cluster_1', 'cluster_2')
  alluvdf <- data.frame('subjectkey'=df[,subjectkey], 
                        'cluster_1'=df[,cluster1], 'cluster_2'=df[,cluster2])
  alluvdf$sex <- df[,sex]
  alluvdf <- alluvdf[order(alluvdf$sex),]
  
  is_alluvia_form(alluvdf)
  ggplot(alluvdf,
         aes(axis1 = cluster_1, axis2 = cluster_2)) +
    geom_alluvium(aes(fill=sex), width = 1/12) +
    geom_stratum(width = 1/12, fill = "black", color = "grey") +
    geom_label(stat = "stratum", infer.label = TRUE) +
    scale_x_discrete(limits = c(cluster1, cluster2), expand = c(.05, .05)) +
    scale_fill_brewer(type = "qual", palette = "Set1") +
    ggtitle(sprintf("%s", plot_title))
}

##########################################################################################
# 
# 
# # Interview period P1
# name_df <- 'imputed_data_P1_ts.csv'
# df <- read.table(file.path(data_path, name_df),
#                  sep=',',
#                  header=TRUE,
#                  as.is=TRUE)
# 
# pdf(paste(plot_path,'/alluvial_P1_ts.pdf', sep=""))
# plotalluvial(df, "Subject relabeling pattern at P1")
# dev.off()
# 
# # Interview period P2
# name_df <- 'imputed_data_P2_ts.csv'
# df <- read.table(file.path(data_path, name_df),
#                  sep=',',
#                  header=TRUE,
#                  as.is=TRUE)
# pdf(paste(plot_path,'/alluvial_P2_ts.pdf', sep=""))
# plotalluvial(df, "Subject relabeling pattern at P2")
# dev.off()
