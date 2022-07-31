require(ComplexHeatmap)
library(RColorBrewer)

# HEATMAPS FOR REPLICABILITY
# Visualize subject distance between train/test and within clusters
# Input: training and test distance matrices, plot title, feature level (either subdomain or domain)

##########################################################################################
work_dir = '/Users/vmandelli/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_strat'
data_path = file.path(work_dir,'results','vero_results')
result_path = file.path(work_dir,'results','vero_results')
plot_path = file.path(work_dir,'plots')
##########################################################################################

replheat <- function(dist_mat_tr, dist_mat_ts, plot_title, level){
  #TRAIN
  if (level=='subdomain'){
    dist_mat_tr$cluster <- dist_mat_tr$cluster_subdomain} else {
      dist_mat_tr$cluster <- dist_mat_tr$cluster_domain
    }
  distdf_tr <- dist_mat_tr[order(apply(subset(dist_mat_tr, select=-grep('cluster', names(dist_mat_tr))), 1, mean)),
                           c(order(apply(subset(dist_mat_tr, select=-grep('cluster', names(dist_mat_tr))), 2, mean)), 
                             ncol(dist_mat_tr))]
  distdf_tr <- distdf_tr[order(distdf_tr$cluster), 
                         c(order(distdf_tr$cluster), 
                           ncol(distdf_tr))]
  clust_tr <- distdf_tr$cluster
  distmat_tr <- as.matrix(subset(distdf_tr, select=-grep('cluster', names(dist_mat_tr))))
  
  row.names(distmat_tr) <- row.names(distdf_tr)
  colnames(distmat_tr) <- names(distdf_tr)[1:(ncol(distdf_tr)-1)]
  
  colSide <- brewer.pal(9, "Set1")[3:9]
  col_v <- list(clusters = c())
  for (idx in sort(unique(clust_tr))){
    col_v$clusters <- c(col_v$clusters, colSide[idx])}
  names(col_v$clusters) <- as.character(sort(unique(clust_tr)))
  
  hTR <- Heatmap(distmat_tr,
                 heatmap_legend_param = list(
                   title = paste('VINELAND', '\ndist mat TR', sep=''), at = seq(min(distmat_tr),
                                                                                max(distmat_tr), 0.5)),
                 # name = paste(name_ins, '\ndist mat TR', sep=''),
                 show_row_names = FALSE,
                 show_column_names = FALSE,
                 show_row_dend = FALSE,
                 show_column_dend = FALSE,
                 cluster_rows = FALSE,
                 cluster_columns = FALSE,
                 # col = colorRampPalette(brewer.pal(8, "Blues"))(25),
                 left_annotation = HeatmapAnnotation(clusters=clust_tr,
                                                     col=col_v, which='row'),
                 top_annotation = HeatmapAnnotation(clusters=clust_tr,
                                                    col=col_v, which='column', 
                                                    show_legend = FALSE))
  # TEST
  if (level=='subdomain'){
    dist_mat_ts$cluster <- dist_mat_ts$cluster_subdomain} else {
      dist_mat_ts$cluster <- dist_mat_ts$cluster_domain
    }
  distdf_ts <- dist_mat_ts[order(apply(subset(dist_mat_ts, select=-grep('cluster', names(dist_mat_ts))), 1, mean)),
                           c(order(apply(subset(dist_mat_ts, select=-grep('cluster', names(dist_mat_ts))), 2, mean)), 
                             ncol(dist_mat_ts))]
  distdf_ts <- distdf_ts[order(distdf_ts$cluster), 
                         c(order(distdf_ts$cluster), 
                           ncol(distdf_ts))]
  clust_ts <- distdf_ts$cluster
  distmat_ts <- as.matrix(subset(distdf_ts, select=-grep('cluster', names(dist_mat_ts))))
  row.names(distmat_ts) <- row.names(distdf_ts)
  colnames(distmat_ts) <- names(distdf_ts)[1:(ncol(distdf_ts)-1)]
  
  col_vts <- list(clusters = c())
  for (idx in sort(unique(clust_ts))){
    col_vts$clusters <- c(col_vts$clusters, colSide[idx])}
  names(col_vts$clusters) <- as.character(sort(unique(clust_ts)))
  
  hTS <- Heatmap(distmat_ts,
                 heatmap_legend_param = list(
                   title = paste('VINELAND', '\ndist mat TS', sep=''), at = seq(min(distmat_ts),
                                                                                max(distmat_ts), 0.5)),
                 # name = paste(name_ins, '\ndist mat TR', sep=''),
                 show_row_names = FALSE,
                 show_column_names = FALSE,
                 show_row_dend = FALSE,
                 show_column_dend = FALSE,
                 cluster_rows = FALSE,
                 cluster_columns = FALSE,
                 # col = colorRampPalette(brewer.pal(8, "Blues"))(25),
                 left_annotation = HeatmapAnnotation(clusters=clust_ts,
                                                     col=col_vts, which='row'),
                 top_annotation = HeatmapAnnotation(clusters=clust_ts,
                                                    col=col_vts, which='column', 
                                                    show_legend = FALSE))
  grid.newpage()
  title = sprintf('%s', plot_title)
  grid.text(title, x=unit(0.5, 'npc'), y=unit(0.8, 'npc'), just='centre')
  pushViewport(viewport(x = 0, y = 0.75, width = 0.5, height = 0.5, just = c("left", "top")))
  grid.rect(gp = gpar(fill = "#00FF0020"))
  draw(hTR, newpage = FALSE)
  popViewport()
  
  pushViewport(viewport(x = 0.5, y = 0.75, width = 0.5, height = 0.5, just = c("left", "top")))
  grid.rect(gp = gpar(fill = "#0000FF20"))
  draw(hTS, newpage = FALSE)
  popViewport()
}

##########################################################################################
datafolder_name <- data_path
# SUBDOMAINS
# P1
name_dmsub_tr_p1 <- 'vineland_distmatsubdomainTRP1.csv'
name_dmsub_ts_p1 <- 'vineland_distmatsubdomainTSP1.csv'

distdfsub_tr_p1 <- read.table(file.path(datafolder_name, 
                                  name_dmsub_tr_p1),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
distdfsub_ts_p1 <- read.table(file.path(datafolder_name, 
                                  name_dmsub_ts_p1),
                        header = TRUE,
                        as.is = TRUE,
                        sep = ',',
                        row.names=1)
### here is the plot!!
pdf(paste(plot_path,'/distmatrix_P1_ts_subdomain.pdf', sep=""))
replheat(distdfsub_tr_p1, distdfsub_ts_p1, '', 'subdomain')
dev.off()


# P2
name_dmsub_tr_p2 <- 'vineland_distmatsubdomainTRP2.csv'
name_dmsub_ts_p2 <- 'vineland_distmatsubdomainTSP2.csv'
distdfsub_tr_p2 <- read.table(file.path(datafolder_name, 
                                        name_dmsub_tr_p2),
                              header = TRUE,
                              as.is = TRUE,
                              sep = ',',
                              row.names=1)
distdfsub_ts_p2 <- read.table(file.path(datafolder_name, 
                                        name_dmsub_ts_p2),
                              header = TRUE,
                              as.is = TRUE,
                              sep = ',',
                              row.names=1)
### here is the plot!!
pdf(paste(plot_path,'/distmatrix_P2_ts_subdomain.pdf', sep=""))
replheat(distdfsub_tr_p2, distdfsub_ts_p2, '', 'subdomain')
dev.off()




# DOMAINS
# P1
name_dmdom_tr_p1 <- 'vineland_distmatdomainTRP1.csv'
name_dmdom_ts_p1 <- 'vineland_distmatdomainTSP1.csv'
distdfdom_tr_p1 <- read.table(file.path(datafolder_name, 
                                        name_dmdom_tr_p1),
                              header = TRUE,
                              as.is = TRUE,
                              sep = ',',
                              row.names=1)
distdfdom_ts_p1 <- read.table(file.path(datafolder_name, 
                                        name_dmdom_ts_p1),
                              header = TRUE,
                              as.is = TRUE,
                              sep = ',',
                              row.names=1)
### here is the plot!!
pdf(paste(plot_path,'/distmatrix_P1_ts_domain.pdf', sep=""))
replheat(distdfdom_tr_p1, distdfdom_ts_p1, '', 'domain')
dev.off()



# P2
name_dmdom_tr_p2 <- 'vineland_distmatdomainTRP2.csv'
name_dmdom_ts_p2 <- 'vineland_distmatdomainTSP2.csv'
distdfdom_tr_p2 <- read.table(file.path(datafolder_name, 
                                        name_dmdom_tr_p2),
                              header = TRUE,
                              as.is = TRUE,
                              sep = ',',
                              row.names=1)
distdfdom_ts_p2 <- read.table(file.path(datafolder_name, 
                                        name_dmdom_ts_p2),
                              header = TRUE,
                              as.is = TRUE,
                              sep = ',',
                              row.names=1)

### here is the plot!!
pdf(paste(plot_path,'/distmatrix_P2_ts_domain.pdf', sep=""))
replheat(distdfdom_tr_p2, distdfdom_ts_p2, '', 'subdomain')
dev.off()


