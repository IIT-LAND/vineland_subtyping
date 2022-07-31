require(ggplot2)
perfdf <- read.table('/Users/ilandi/PycharmProjects/ndar-stratification/out/gridsearch_perf_ptype2_p1p2.csv',
                   as.is=TRUE,
                   sep=';',
                   header=TRUE)
perfdf$perc_na <- as.factor(perfdf$na_perc_thrs)
perfdf$group1 <- paste(perfdf$period, perfdf$feat_lev, sep='-')
ci <- lapply(sapply(perfdf$val_ci, function(x){strsplit(x, ',')}), 
             function(x){c(as.numeric(gsub('\\(', '', x[1])), as.numeric(gsub('\\)', '', x[2])))})
low <- c()
high <- c()
for (lab in ci){
  low <- c(low, lab[1])
  high <- c(high, lab[2])
}
perfdf$low <- low
perfdf$high <- high

for (cv in c(2, 5, 10)){
  perf <- perfdf[perfdf$cv_fold==cv,]
  sp <- ggplot(data = perf, aes(x = na_perc_thrs, y = val_acc, group = group1))
  print(sp + geom_line(size=1, aes(colour=period, linetype=feat_lev)) +
    facet_grid(~ n_neigh) +
    # geom_errorbar(
    #   data = perf,
    #   aes(perc_na, val_acc, ymin = low, ymax = high),
    #   width = 0.4
    # )
    geom_point(data=perf, aes(na_perc_thrs, test_acc, group=group1, colour=period, shape=feat_lev), size=3) +
    guides(colour=guide_legend(order=1, override.aes=list(shape=c(NA, NA), linetype=c(1, 1))),
           linetype=guide_legend(order=2, override.aes=list(shape=c(NA, NA), linetype=c(1, 2), size=0.5))) +
    labs(col='Age period', linetype='Feature levels (validation set)', shape='Feature levels (test set)') +
    ggtitle(sprintf('Accuracy performance on validation and test sets CV-%d', cv)) +
    theme(plot.title = element_text(hjust = 0.5)))}
  # aes(colour=age_period, shape=feat_level) + scale_shape_manual(values = c('domains' = 16, 'subdomains' = 17))
