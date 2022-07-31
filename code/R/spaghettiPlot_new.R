# function to compute lme and make spaghetti plot
spaghettiPlot_new <- function(df, x_var, y_var, subgrp_var, xLabel, yLabel,
                          modelType = "quadratic",
                          fname2save = NULL, plot_dots = TRUE, plot_lines = TRUE,
                          ci_band = TRUE, pi_band = FALSE, dot_alpha = 1/10, line_alpha = 1/10,
                          band_alpha = 1/2, xLimits = c(-5, 60), yLimits = NULL, standardize = FALSE, color = c("#FF6666","#99CC66","#99FFFF","#E7E1Ef")) {
    
    
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
    # xLabel = label to put on x-axis
    # yLabel = label to put on y-axis
    # fname2save = file path and filename if you want to save plot
    # plot_dots = set to TRUE to plot individual dots
    # plot_lines = set to TRUE to plot individual lines
    # ci_band = set to true if you want to plot confidence bands
    # pi_band = set to true if you want to plot prediction bands
    # dot_alpha = alpha setting for plotting dots
    # line_alpha = alpha setting for plotting lines
    # band_alpha = alpha setting for plotting bands
    # xLimits = set limits on x-axis
    # yLimits = set limits on y-axis
    # standardize = set to TRUE if you want to z-score data before modeling
    #
    # OUTPUT
    # a list with the plot information p and the linear mixed effect model
    #
    
    # make sure the required libraries are loaded
    require(ggplot2)
    require(nlme)
    
    # if you are plotting prediction bands, turn off confidence bands
    if (pi_band) {
        ci_band = FALSE
    }# if (pi_band)
    
    # find unique subgroups and grab them from input data frame
    subgrps = df[,subgrp_var]
    unique_subgrps = unique(subgrps)
    unique_subgrps = unique_subgrps[!unique_subgrps=="NA"]
    #unique_subgrps = unique_subgrps$subgrp[!unique_subgrps$subgrp=="NA"]
    subgrp_mask = is.element(df[,subgrp_var],unique_subgrps)
    df2use = df[subgrp_mask,]
    #df2use = subset(df,is.element(df[[subgrp_var]],unique_subgrps))
    # uniqueSubs = unique(df2use$subjectId)
    
    #------------------------------------------------------------------------------
    # initialize the plot
    p = ggplot(data = df2use, aes(x = get(x_var), y = get(y_var), group = subjectId))
    
    #------------------------------------------------------------------------------
    # plot individual lines
    if (plot_lines) {
        p = p + geom_line(aes(colour = get(subgrp_var)), alpha = line_alpha) + guides(alpha = FALSE) 
        p = p + scale_colour_manual(values=color)
    }# if (plot_lines)
    
    #------------------------------------------------------------------------------
    # plot individual dots
    if (plot_dots) {
        # plot each data point as a dot
        # p = p + geom_point(aes(colour = get(subgrp_var), alpha = dot_alpha)) + guides(alpha=FALSE)
        p = p + geom_point(aes(colour = get(subgrp_var)), alpha = dot_alpha,size=1) + guides(alpha=FALSE)
      
    } # if (plot_dots)#
    
    #------------------------------------------------------------------------------
    
    if (modelType=="linear") {
        # compute linear mixed effect model--------------------------------------------
        fx_form = as.formula(sprintf("%s ~ %s*%s", y_var, x_var, subgrp_var))
        rx_form = as.formula(sprintf("~ %s|subjectId",x_var))
    } else if (modelType=="quadratic") {
        # compute quadratic mixed effect model----------------------------------------
        fx_form = as.formula(sprintf("%s ~ %s + I(%s^2) + %s + %s:%s + I(%s^2):%s",
                                     y_var, x_var, x_var, subgrp_var, x_var, subgrp_var, x_var, subgrp_var))
        # rx_form = as.formula(sprintf("~ %s|subjectId + I(%s^2)|subjectId",x_var, x_var))
        # rx_form = as.formula(sprintf("~ I(%s^2)|subjectId",x_var))
        rx_form = as.formula(sprintf("~ %s|subjectId",x_var))
    }# if (modelType=="linear")
    
    ctrl <- lmeControl(opt='optim', msMaxIter = 500)
    ml_method = "ML"
    m_allgrps <- eval(substitute(lme(fixed = fx_form, random = rx_form, data = df2use,
                                     na.action = na.omit, control=ctrl, method = ml_method), list(fx_form = fx_form, rx_form = rx_form)))
    # summary(m_allgrps)
    # anova(m_allgrps)
    
    #------------------------------------------------------------------------------
    # get information for confidence and prediction bands
    newdat = expand.grid(x = min(df2use[,x_var],na.rm = TRUE):max(df2use[,x_var],na.rm = TRUE), s=sort(unique_subgrps))
    colnames(newdat)[1] = x_var
    colnames(newdat)[2] = subgrp_var
    newdat[,2] = as.factor(newdat[,2])
    newdat$pred <- predict(m_allgrps, newdat, level = 0)
    colnames(newdat)[3] = y_var
    Designmat <- model.matrix(formula(m_allgrps)[-2], newdat)
    predvar <- diag(Designmat %*% vcov(m_allgrps) %*% t(Designmat))
    newdat$SE <- sqrt(predvar)
    newdat$SE2 <- sqrt(predvar+m_allgrps$sigma^2)
    # newdat$subjectId = newdat$subgrpDx
    newdat$subjectId = newdat[,2]
    
    #------------------------------------------------------------------------------
    # plot confidence or prediction bands
    p1 = ggplot(data = df2use, aes(x = get(x_var), y = get(y_var), group = subjectId)) ### add here
    if (ci_band) {
        # plot lme line and 95% confidence bands
      p1 = p1 + scale_colour_manual(values=color)  
      p1 = p1+  geom_ribbon(aes(x = get(x_var),y = get(y_var),ymin=get(y_var)-2*SE,ymax=get(y_var)+2*SE,
                                colour = get(subgrp_var), fill = get(subgrp_var)),data = newdat, alpha = band_alpha) 
        p1 = p1 +  scale_fill_manual(values=color)
        p1 = p1 + geom_line(aes(x = get(x_var), y = get(y_var), colour = get(subgrp_var)),data = newdat)
        
    } else if (pi_band)	{
        # plot lme line and 95% prediction bands
        p1 = p1+  geom_ribbon(aes(x = get(x_var),y = get(y_var),ymin=get(y_var)-2*SE2,ymax=get(y_var)+2*SE2,
                                colour = get(subgrp_var), fill = get(subgrp_var)),data = newdat, alpha = band_alpha) +
            geom_line(aes(x = get(x_var), y = get(y_var), colour = get(subgrp_var)),data = newdat)
    }# if (ci_band)
    
    # change y limits if necessary
    if (!is.null(yLimits)) {
        p = p + scale_y_continuous(limits = yLimits)
        p1 = p1 + scale_y_continuous(limits = yLimits)
    }
    
    p = p + theme(legend.title=element_blank()) + xlab(xLabel) + ylab(yLabel) +
        scale_x_continuous(limits = xLimits)+ theme_bw()
    p1 = p1 + theme(legend.title=element_blank()) + xlab(xLabel) + ylab(yLabel) +
        scale_x_continuous(limits = xLimits)+ theme_bw()
    #------------------------------------------------------------------------------
    # save plot
    if (!is.null(fname2save)) {
        ggsave(filename = fname2save)
    }# if
    
    #------------------------------------------------------------------------------
    # standardize data
    if (standardize){
        df2use[,y_var] = (df2use[,y_var] - mean(df2use[,y_var], na.rm=TRUE))/sd(df2use[,y_var], na.rm = TRUE)
        df2use[,x_var] = (df2use[,x_var] - mean(df2use[,x_var], na.rm = TRUE))/sd(df2use[,x_var], na.rm = TRUE)
        
        # re-run model on mean centered data
        m_allgrps <- eval(substitute(lme(fixed = fx_form, random = rx_form, data = df2use,
                                         na.action = na.omit, control=ctrl, method = ml_method),
                                     list(fx_form = fx_form, rx_form = rx_form)))
    }
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # output information
    p
    p1
    result = list(p = p, p1 = p1, lme_model = m_allgrps)
    
} 

spaghettiPlot_new
#==============================================================================


