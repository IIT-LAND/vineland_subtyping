import numpy as np
import pandas as pd
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, \
    ColorBar, HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import export_png
from math import pi
from matplotlib import pyplot as plt
import math


def plot_metrics(cv_score, title, figsize=(10, 5), save_fig=None):
    """
   
    Function that plot the average performance (i.e., normalized stability) over cross-validation
    for training and validation sets.

    Parameters
    ----------
    :param cv_score: reval performance output
    :type cv_score: dictionary
    :param title: figure title
    :type title: str
    :param figsize: performance metric figure size
    :type figsize: tuple (width, height)
    :param save_fig: name of the figure to save (without format), default None
    :type save_fig: str
   
    Function that plots the average performance (i.e., normalized stability) over cross-validation
    for training and validation sets. The horizontal lines represent the random performance error
    for the correspondent number of clusters.
    :param cv_score: collection of cv scores as output by `reval.best_nclust_cv.FindBestCLustCV.best_nclust`.
    :type cv_score: dictionary
    :param figsize: (width, height), default (8, 5).
    :type figsize: tuple
    :param linewidth: width of the lines to draw.
    :type linewidth: int
    :param color: line colors for train and validation sets, default ('black', 'black').
    :type color: tuple
    :param legend_loc: legend location, default 2.
    :type legend_loc: int
    :param fontsize: size of fonts, default 12.
    :type fontsize: int
    :param title: figure title, default "".
    :type title: str
    :param prob_lines: plot the normalized stability of random labeling as thresholds, default False.
    :type prob_lines: bool
    :param save_fig: file name for saving figure in png format, default None.
    :type save_fig: str
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(cv_score['train'].keys()),
            [me[0] for me in cv_score['train'].values()],
            linewidth=5,
            label='training set')
    ax.errorbar(list(cv_score['val'].keys()),
                [me[0] for me in cv_score['val'].values()],
                [me[1][1] for me in cv_score['val'].values()],
                linewidth=5,
                label='validation set')
    ax.legend(fontsize=18, loc=1)
    plt.xticks([lab for lab in cv_score['train'].keys()], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of clusters', fontsize=18)
    plt.ylabel('Normalized stability', fontsize=18)
    plt.title(title, fontsize=18)
    if save_fig is not None:
        plt.savefig(f'./{save_fig}', format='png')
    else:
        plt.show()

    

def plot_miss_heat(df_ts, cl_labels, feat, values, period, hierarchy, save_fig=None):
    heat_df = pd.DataFrame({'cl_labels': cl_labels, 'feat': feat, 'values': values})

    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
              "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
              "#550b1d"][::-1]

    mapper = LinearColorMapper(palette=colors,
                               low=0,
                               high=100)
    p = figure(x_range=[c for c in df_ts.columns.intersection(np.unique(feat))],
               y_range=[str(lab) for lab in sorted(np.unique(cl_labels))],
               x_axis_location="above",
               plot_width=1200,
               plot_height=600,
               toolbar_location='below',
               title=f'Percentage of originally missing information by subcluster '
                     f'for each feature {hierarchy} of the Vineland TEST dataset at period {period}')

    TOOLTIPS = [('score', '@values')]

    p.add_tools(HoverTool(tooltips=TOOLTIPS))

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_text_font_size = "15pt"
    p.yaxis.major_label_text_font_size = "15pt"
    p.title.text_font_size = '13pt'
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 4

    p.rect(x="feat", y="cl_labels",
           width=1, height=1,
           source=heat_df,
           fill_color={'field': 'values',
                       'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%.2f"),
                         label_standoff=8, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    if save_fig is not None:
        export_png(p, filename=f'./plot/{save_fig}.png')
    else:
        show(p)


"""
Private functions
"""


def _scatter_plot(umap_mtx,
                  pid_subc_list,
                  colors,
                  fig_height,
                  fig_width,
                  label,
                  title='',
                  save_fig=None):
    """
    Bokeh scatterplot to visualize in jupyter clusters and subject info.

    :param umap_mtx: Array with UMAP projections
    :type umap_mtx: numpy array
    :param pid_subc_list: list of pids ordered as in umap_mtx and subcluster labels
    :type pid_subc_list: list of tuples
    :param colors: Color list
    :type colors: list
    :param fig_height: figure height
    :type fig_height: int
    :param fig_width: figure width
    :type fig_width: int
    :param label: dictionary of class numbers and subtype labels
    :type label: dict
    :param title: figure title
    :type title: str
    :param save_fig: flag to enable figure saving, defaults to None
    :type save_fig: str
    """

    pid_list = list(map(lambda x: x[0], pid_subc_list))
    subc_list = list(map(lambda x: x[1], pid_subc_list))
    df_dict = {'x': umap_mtx[:, 0].tolist(),
               'y': umap_mtx[:, 1].tolist(),
               'pid_list': pid_list,
               'subc_list': subc_list}

    df = pd.DataFrame(df_dict).sort_values('subc_list')

    source = ColumnDataSource(dict(
        x=df['x'].tolist(),
        y=df['y'].tolist(),
        pid=df['pid_list'].tolist(),
        subc=list(map(lambda x: label[str(x)], df['subc_list'].tolist())),
        col_class=[str(i) for i in df['subc_list'].tolist()]))

    labels = [str(i) for i in df['subc_list']]
    cmap = CategoricalColorMapper(factors=sorted(pd.unique(labels)),
                                  palette=colors)
    TOOLTIPS = [('pid', '@pid'),
                ('subc', '@subc')]

    plotTools = 'box_zoom, wheel_zoom, pan,  crosshair, reset, save'

    output_notebook()
    p = figure(plot_width=fig_width * 80, plot_height=fig_height * 80,
               tools=plotTools, title=title)
    p.add_tools(HoverTool(tooltips=TOOLTIPS))
    p.circle('x', 'y', legend_group='subc', source=source,
             color={'field': 'col_class',
                    "transform": cmap}, size=12)
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_color = None
    p.yaxis.major_label_text_color = None
    p.grid.grid_line_color = None
    p.title.text_font_size = '13pt'
    p.legend.label_text_font_size = '18pt'
    p.legend.location = 'top_left'
    if save_fig is not None:
        export_png(p, filename=f'./plot/{save_fig}.png')
    else:
        show(p)


def _confint(vect):
    """
    Parameters
    ----------
    vect: list (of performance scores)
    Returns
    ------
    float: value to +/- to stability error for error bars (95% CI)

    """
    error = np.mean(vect)
    return 1.96 * math.sqrt((error * (1 - error)) / len(vect))
