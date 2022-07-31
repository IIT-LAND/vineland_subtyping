# function to compute lme and make spaghetti plot
spaghettiPlot2_nodataset <- function(data2use, x_var, y_var, subgrp_var, cov_vars=NULL, xLabel, yLabel,
                           modelType = "quadratic",
                           fname2save = NULL, plot_dots = TRUE, plot_lines = TRUE,
                           dot_alpha = 1/10, line_alpha = 1/10,
                           ci_band = TRUE, pi_band = FALSE, band_alpha = 1/2,
                           xLimits = c(-5, 60), yLimits = NULL,
                           lineColor = "subgrp", dotColor = "subgrp",
                           ml_method = "ML",
                           legend_name = "Slope", maxIter = 500, plot_title) {
  # DESCRIPTION
  # This function takes a data frame, and strings denoting the x, y, and
  # subgroup variables to plot.  It also computes the linear mixed effect
  # model on the data.
  #
  # INPUT ARGUMENTS
  # df = data frame to process
  # x_var = string denoting variable to plot on x-axis
  # y_var = string denoting variable to plot on y-axis
  # subgrp_var = string denoting variable that has subgroup labels
  # cov_vars = string to denote covariates to include in the model (leave as NULL if no covariates)
  # xLabel = label to put on x-axis
  # yLabel = label to put on y-axis
  # fname2save = file path and filename if you want to save plot
  # plot_dots = set to TRUE to plot individual dots
  # plot_lines = set to TRUE to plot individual lines
  # ci_band = set to true if you want to plot confidence bands
  # pi_band = set to true if you want to plot prediction bands
  # dot_alpha = alpha setting for plotting dots
  # line_alpha = alpha setting for plotting lines
  #
  # OUTPUT
  # a list with the plot information p and the linear mixed effect model
  #

  # make sure the required libraries are loaded
  require(ggplot2)
  require(lmerTest)
  require(visreg)
  require(nlme)

  # find unique subgroups and grab them from input data frame
  subgrps = data2use[,subgrp_var]
  unique_subgrps = unique(subgrps)
  unique_subgrps = unique_subgrps[!unique_subgrps=="NA"]
  #unique_subgrps = unique_subgrps$subgrp[!unique_subgrps$subgrp=="NA"]
  subgrp_mask = is.element(data2use[,subgrp_var],unique_subgrps)
  df2use = data2use[subgrp_mask,]
  #df2use = subset(data2use,is.element(data2use[[subgrp_var]],unique_subgrps))
  # uniqueSubs = unique(df2use$subjectId)

  #------------------------------------------------------------------------------
  # initialize the plot
  p = ggplot(data = df2use, aes(x = get(x_var), y = get(y_var), group = subjectId))

  #------------------------------------------------------------------------------
  # plot individual lines
  if (plot_lines) {
    if (lineColor=="subgrp"){
      p = p + geom_line(aes(colour = get(subgrp_var)), alpha = line_alpha) + guides(alpha = FALSE)
    } else if (lineColor=="subjectId"){
      p = p + geom_line(aes(colour = subjectId), alpha = line_alpha) + guides(alpha = FALSE,colour=FALSE)
    } else if (lineColor=="rfx_slope"){
      cols2use = colorRampPalette(c("blue","red"))(100)
      p = p + geom_line(aes(colour = get(lineColor)), alpha = line_alpha) + guides(alpha = FALSE) +
        scale_color_gradientn(colors=cols2use, name=legend_name)
    } else if (lineColor=="centiles"){
      cols2use = colorRampPalette(c("red","blue"))(length(unique(data2use$centiles)))
      p = p + geom_line(aes(colour = get(subgrp_var)), alpha = line_alpha) + guides(alpha = FALSE) +
        scale_color_manual(values=cols2use, name=legend_name)
    } else{
      p = p + geom_line(aes(colour = get(lineColor)), alpha = line_alpha) + guides(alpha = FALSE)
    }
  }# if (plot_lines)

  #------------------------------------------------------------------------------
  # plot individual dots
  if (plot_dots) {
    if (dotColor=="subgrp"){
      # plot each data point as a dot
      # p = p + geom_point(aes(colour = get(subgrp_var), alpha = dot_alpha)) + guides(alpha=FALSE)
      p = p + geom_point(aes(colour = get(subgrp_var)), alpha = dot_alpha) + guides(alpha=FALSE)
    } else if (dotColor=="subjectId"){
      p = p + geom_point(aes(colour = subjectId), alpha = dot_alpha) + guides(alpha=FALSE,colour=FALSE)
    } else if (lineColor=="rfx_slope"){
      cols2use = colorRampPalette(c("blue","red"))(100)
      p = p + geom_point(aes(colour = get(dotColor)), alpha = line_alpha) + guides(alpha = FALSE) +
        scale_color_gradientn(colors=cols2use, name=legend_name)
    } else if (dotColor=="centiles"){
      cols2use = colorRampPalette(c("red","blue"))(length(unique(data2use$centiles)))
      p = p + geom_point(aes(colour = get(subgrp_var)), alpha = dot_alpha) + guides(alpha=FALSE) +
        scale_color_manual(values=cols2use, name=legend_name)
    } else{
      p = p + geom_point(aes(colour = get(dotColor)), alpha = dot_alpha) + guides(alpha=FALSE)
    }
  } # if (plot_dots)

  #------------------------------------------------------------------------------
  # covariates ***NEW*** ------------------------------------------------------
  if (!is.null(cov_vars)){
    for (icov in 1:length(cov_vars)){
      if (icov==1){
        covs2use = sprintf("+ %s",cov_vars[icov])
      } else {
        covs2use = sprintf("%s + %s", covs2use, cov_vars[icov])
      } # if (icov==1){
    } # for (icov in 1:length(cov_vars)){
  } else {
    covs2use = ""
  } # if (is.null(cov_vars)){

  #------------------------------------------------------------------------------
  if (modelType=="linear") {
    form2use = as.formula(sprintf("%s ~ %s*%s%s + (%s|subjectId)",                   #+ (1|%s) +
                                  y_var, x_var, subgrp_var, covs2use, x_var))        #"dataset", 
  } else if (modelType=="quadratic") {
    form2use = as.formula(sprintf("%s ~ %s + I(%s^2) + %s + %s:%s + I(%s^2):%s%s +  (%s|subjectId)", #(1|%s) +
                                  y_var, x_var, x_var, subgrp_var, x_var, subgrp_var, x_var, subgrp_var, covs2use,  x_var))  #"dataset",
  }# if (modelType=="linear")

  # ctrl <- lmeControl(opt='optim', msMaxIter = maxIter)
  # ctrl = lmerControl(optCtrl=list(maxfun=maxIter))
  ctrl = lmerControl(optCtrl=list(maxfun=maxIter), # lmerControl(optCtrl=list(maxfun=maxIter)
                     check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4),check.nobs.vs.nRE = "ignore")

  # ml_method = ml_method
  m_allgrps <- eval(substitute(lmer(formula = form2use, data = df2use, na.action = na.omit, control = ctrl)))
  # summary(m_allgrps)
  # anova(m_allgrps)

  # # use visreg to make the fit plots
  # if (subgrp_var=="centiles"){
  #   colors2use = colorRampPalette(c("red","blue"))(length(unique(data2use$centiles)))
  #   # p2 = visreg(m_allgrps, x_var, by = subgrp_var, band=TRUE, partial=FALSE, rug=FALSE, gg=TRUE, overlay=TRUE) +
  #   #   scale_colour_manual(values = colors2use) +
  #   #   scale_fill_manual(values = colors2use) +
  #   #   xlab(xLabel) + ylab(yLabel) #+
  #   #   #scale_x_continuous(limits = xLimits) + ylim(yLimits)
  # }else {
  #   colors2use = get_ggColorHue(n = length(unique_subgrps))
  #   # p2 = visreg(m_allgrps, x_var, by = subgrp_var, band=TRUE, partial=FALSE, rug=FALSE, gg=TRUE, overlay=TRUE) +
  #   #   scale_colour_manual(values = colors2use) +
  #   #   scale_fill_manual(values = colors2use) +
  #   #   xlab(xLabel) + ylab(yLabel) #+
  #   #   #scale_x_continuous(limits = xLimits) + ylim(yLimits)
  # }
  #
  # p2 = visreg(m_allgrps, x_var, by = subgrp_var, band=TRUE, partial=FALSE, rug=FALSE, gg=TRUE, overlay=TRUE) +
  #   scale_colour_manual(values = colors2use) +
  #   scale_fill_manual(values = colors2use) +
  #   xlab(xLabel) + ylab(yLabel)
  #
  # if (!is.null(xLimits)) {
  #   p2 = p2 + scale_x_continuous(limits = xLimits)
  # }
  # if (!is.null(yLimits)) {
  #   p2 = p2 + scale_y_continuous(limits = yLimits)
  # }
  # p2 = p2 + ggtitle(plot_title) + theme(plot.title = element_text(hjust = 0.5))





  #------------------------------------------------------------------------------
  if (modelType=="linear") {
    form2use = as.formula(sprintf("%s ~ %s*%s%s + (%s|subjectId)", #(1|%s) + 
                                  y_var, x_var, subgrp_var, covs2use,  x_var))# "dataset",
  } else if (modelType=="quadratic") {
    form2use = as.formula(sprintf("%s ~ %s + I(%s^2) + %s + %s:%s + I(%s^2):%s%s +  (%s|subjectId)", #(1|%s) +
                                  y_var, x_var, x_var, subgrp_var, x_var, subgrp_var, x_var, subgrp_var, covs2use,x_var)) # "dataset", 
  }# if (modelType=="linear")



  #------------------------------------------------------------------------------
  # model for the plot ***NEW*** ----------------------------------------------
  if (modelType=="linear") {
    # compute linear mixed effect model--------------------------------------------
    fx_form = as.formula(sprintf("%s ~ %s*%s", y_var, x_var, subgrp_var))
    # rx_form = as.formula(sprintf("~ %s|subjectId",x_var))
    rx_form = as.formula(sprintf("~ + %s|subjectId",x_var)) #1|%s #"dataset",
  } else if (modelType=="quadratic") {
    # compute quadratic mixed effect model----------------------------------------
    fx_form = as.formula(sprintf("%s ~ %s + I(%s^2) + %s + %s:%s + I(%s^2):%s",
                                 y_var, x_var, x_var, subgrp_var, x_var, subgrp_var, x_var, subgrp_var))
    # rx_form = as.formula(sprintf("~ %s|subjectId",x_var))
    rx_form = as.formula(sprintf("~ + %s|subjectId",x_var)) #1|%s  "dataset",
  }# if (modelType=="linear")

  ctrl <- lmeControl(opt='optim', msMaxIter = maxIter)
  ml_method = ml_method
  m_allgrps_plot <- eval(substitute(lme(fixed = fx_form, random = rx_form, data = df2use,
                                        na.action = na.omit, control=ctrl, method = ml_method), list(fx_form = fx_form, rx_form = rx_form)))

  #------------------------------------------------------------------------------
  # get information for confidence and prediction bands
  newdat = expand.grid(x = min(df2use[,x_var],na.rm = TRUE):max(df2use[,x_var],na.rm = TRUE), s=sort(unique_subgrps))
  colnames(newdat)[1] = x_var
  colnames(newdat)[2] = subgrp_var
  newdat[,2] = as.factor(newdat[,2])
  newdat$pred <- predict(m_allgrps_plot, newdat, level = 0)
  # newdat$pred <- predict(m_allgrps, newdat, level = 0)

  colnames(newdat)[3] = y_var

  # Designmat <- model.matrix(formula(m_allgrps)[-2], newdat)
  # predvar <- diag(Designmat %*% vcov(m_allgrps) %*% t(Designmat))
  # newdat$SE <- sqrt(predvar)
  # newdat$SE2 <- sqrt(predvar+m_allgrps$sigma^2)
  # # newdat$subjectId = newdat$subgrpDx
  # newdat$subjectId = newdat[,2]
  # ***NEW*** -----------------------------------------------------------------
  Designmat <- model.matrix(formula(m_allgrps_plot)[-2], newdat)
  predvar <- diag(Designmat %*% vcov(m_allgrps_plot) %*% t(Designmat))
  newdat$SE <- sqrt(predvar)
  newdat$SE2 <- sqrt(predvar+m_allgrps_plot$sigma^2)
  # newdat$subjectId = newdat$subgrpDx
  newdat$subjectId = newdat[,2]

  #------------------------------------------------------------------------------
  # plot confidence or prediction bands
  if (ci_band) {
    # plot lme line and 95% confidence bands
    p = p + geom_ribbon(aes(x = get(x_var),y = get(y_var),ymin=get(y_var)-2*SE,ymax=get(y_var)+2*SE,
                            colour = get(subgrp_var), fill = get(subgrp_var)),data = newdat, alpha = band_alpha) +
      geom_line(aes(x = get(x_var), y = get(y_var), colour = get(subgrp_var)),data = newdat)
  } else if (pi_band)	{
    # plot lme line and 95% prediction bands
    p = p + geom_ribbon(aes(x = get(x_var),y = get(y_var),ymin=get(y_var)-2*SE2,ymax=get(y_var)+2*SE2,
                            colour = get(subgrp_var), fill = get(subgrp_var)),data = newdat, alpha = band_alpha) +
      geom_line(aes(x = get(x_var), y = get(y_var), colour = get(subgrp_var)),data = newdat)
  }# if (ci_band)



  # change y limits if necessary
  if (!is.null(yLimits)) {
    p = p + scale_y_continuous(limits = yLimits)
  }

  if (!is.null(xLimits)) {
    p = p + scale_x_continuous(limits = xLimits)
  }

  p = p + theme(legend.title=element_blank()) + xlab(xLabel) + ylab(yLabel) #+
    #scale_x_continuous(limits = xLimits)

  p = p + ggtitle(plot_title) + theme(plot.title = element_text(hjust = 0.5))

  #------------------------------------------------------------------------------
  # save plot
  if (!is.null(fname2save)) {
    ggsave(filename = fname2save)
  }# if

  #------------------------------------------------------------------------------
  # output information
  p
  # result = list(p = p, p2 = p2, lme_model = m_allgrps)
  result = list(p = p, lme_model = m_allgrps)
  return(result)
} # spaghettiPlot2
