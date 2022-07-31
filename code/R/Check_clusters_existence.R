## check clusters existence
library(sigclust)
library(dplyr)
library(ggplot2)
library(wesanderson)

# VABS data
data <- read.csv("~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/results/FINAL_correct/ndar_withumap.csv", header=TRUE)
color =  wes_palette("IsleofDogs1")
# set option
nreps_sim = 10000
nreps_sigclust_repeat = 10000

data$cluster_domain =factor(data$cluster_domain, levels = c("2",'3','1'))
# run sigclust on train
data2use = data %>% filter(TR_TS=="tr") %>% select(UMAP_1,UMAP_2,cluster_domain)
rseo_sigclust_res_train = sigclust(data2use[,c("UMAP_1","UMAP_2")],
                             nsim = nreps_sim, nrep = nreps_sigclust_repeat,
                             labflag = 1, 
                             label = data2use[,"cluster_domain"],
                             icovest=2)
p_train = ggplot(data=data2use, aes(x=UMAP_1,y=UMAP_2)) +
    geom_density_2d_filled() + 
    geom_point(data=data2use, aes(colour=cluster_domain), alpha = 0.5,size = 0.5)+ggtitle('train')+ 
    scale_y_continuous( limits=c(-5, 1)) + scale_x_continuous( limits=c(-5,8))+ scale_colour_manual(values = color)
p_train
rseo_sigclust_res_train
p_val_train = (sum(rseo_sigclust_res_train@simcindex<=rseo_sigclust_res_train@xcindex)+1)/(nreps_sim+1)
print(p_val_train)

# run sigclust on test
data2use = data %>% filter(TR_TS=="ts") %>% select(UMAP_1,UMAP_2,cluster_domain)
rseo_sigclust_res_test = sigclust(data2use[,c("UMAP_1","UMAP_2")],
                             nsim = nreps_sim, nrep = nreps_sigclust_repeat,
                             labflag = 1, 
                             label = data2use[,"cluster_domain"],
                             icovest=2)
p_test = ggplot(data=data2use, aes(x=UMAP_1,y=UMAP_2)) +
    geom_density_2d_filled() + 
    geom_point(data=data2use, aes(colour=cluster_domain), alpha = 0.5,size = 0.5)+ggtitle('validation')+ 
    scale_y_continuous( limits=c(-5, 1)) + scale_x_continuous( limits=c(-5, 8))+ scale_colour_manual(values = color)
p_test
rseo_sigclust_res_test
p_val_test = (sum(rseo_sigclust_res_test@simcindex<=rseo_sigclust_res_test@xcindex)+1)/(nreps_sim+1)
print(p_val_test)

library(ggpubr)
umap_tr_ts = ggarrange(p_train, p_test, nrow=1, common.legend = TRUE, legend="bottom")
umap_tr_ts
plot_path = '~/OneDrive - Fondazione Istituto Italiano Tecnologia/vineland_proj_new/plots/FINAL_correct'
ggsave(file.path(plot_path,'Clusters_significance.png'), umap_tr_ts, width = 7 ,height = 4)

