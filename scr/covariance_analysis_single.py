#!/usr/bin/env python2.7

from plot_modules import *
import seaborn as sns
import matplotlib.pylab as pylab

#sns.set_style("whitegrid")

params = {'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

fdir_plot = './plots/'


def make_xyz(ndata=800):

    import numpy.random as npr
    import numpy as np

    x = np.linspace(-5.0,5.0,ndata)
    ran1 = npr.normal(0.0,0.5,ndata)
    ran2 = npr.normal(0.0,0.1,ndata)
    y = x*1.2 - 0.4 + ran1 + ran2
    z = x*0.8 + 0.2 + ran2
    
    return x , y, z


def plot_caller(func, data, xline, var_num=2, labels=None, title=True,
                white=False, xlog=True, ylog=True, legend=True, **kwargs):

    if white:
        sns.set_style("white")
        plt.grid(True)
    else:
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    if kwargs['nrow'] == 1:
        ax = plt.gca()
    else:
        fig, ax = plt.subplots(nrows=kwargs['nrow'], sharex=True, sharey=False)

    colors = ["#67E568", "#257F27", "#08420D", "#FFF000", "#FFB62B", "#E56124", "#E53E30", "#7F2353", "#F911FF",
              "#9F8CA6"]
    colors = ["#08420D", "#257F27", "#67E568", "#FFB62B",
              "#E56124", "#E53E30", "#7F2353", "#F911FF",
              "#9F8CA6"]
    # colors = ["#24a8e5", "#FFB62B", "#E53E30", "#7F2353", "#F911FF", "#9F8CA6"]

    if var_num == 3:
        for ir in range(len(labels)):
            func(ax, data[0][ir], data[1][ir], data[2][ir],
                 xline[ir], labels[ir], ir,
                 marker='o', color=colors[ir], **kwargs)
    if var_num == 2:
        for ir in range(len(labels)):
            func(ax, data[0][ir], data[1][ir], xline[ir], labels[ir], ir,
                 marker='o', color=colors[ir], **kwargs)
    if var_num == 'report':
        for ir in range(len(labels)):
            func(data[0][ir], data[1][ir], data[2][ir], labels[ir], ir, **kwargs)
        return

    # plt.errorbar(14.7, -0.69, yerr=0.02, fmt='o', capthick=2, label='Rhapsody-G')

    fdir_plot = './plots/'
    fname = kwargs['fname']+'_'+kwargs['delta']+'_'+kwargs['filter_type']+'.png'

    try :
        for axi in ax:
            axi.set(adjustable='box-forced')#, aspect='equal')

        if xlog:
            ax[1].set_xlabel(r'$\log($%s$_{\Delta}$'%kwargs['xlabel']+' '+kwargs['xunit']+r'$)$', size=18)
        else:
            ax[1].set_xlabel(r'%s$_{\Delta}$'%kwargs['xlabel']+' '+kwargs['xunit'], size=18)
        ax[0].set_ylabel('Slope', size=18)
        ax[1].set_ylabel(r'scatter', size=18)
        if title:
            ax[0].set_title(r'$\Delta = %s$   %s'%(kwargs['delta'],kwargs['ylabel']), size=20)
        else:
            ax[0].set_title(' ')

        ax[1].set_xlim(kwargs['xlim'])
        for i in range(kwargs['nrow']):
            ax[i].set_ylim(kwargs['ylim'][i])
    except TypeError:
        plt.xlim(kwargs['xlim'])
        plt.ylim(kwargs['ylim'])
        if xlog:
            plt.xlabel(r'$\log($'+kwargs['xlabel']+' '+kwargs['xunit']+r'$)$', size=18)
        else:
            plt.xlabel(kwargs['xlabel'] + ' ' + kwargs['xunit'], size = 18)
        if ylog:
            plt.ylabel(r'$\log($'+kwargs['ylabel']+' '+kwargs['yunit']+r'$)$', size=18)
        else:
            plt.ylabel(kwargs['ylabel']+' '+kwargs['yunit'], size=18)
        if title:
            ax.set_title(r'$\Delta = %s$'%kwargs['delta'], size=20)
        else:
            ax.set_title(' ')
        if legend:
            legend = ax.legend(loc=kwargs['legend_loc'], fontsize=kwargs['legend_fontsize'],
                               title=kwargs['legend_title'], fancybox=True, shadow=True)

        try:
            frame = legend.get_frame()
            legend.set_frame_on(True)
            frame.set_facecolor('white')
            plt.setp(legend.get_title(), fontsize=kwargs['legend_fontsize'])
        except AttributeError:
            pass

    plt.savefig(fdir_plot+fname, bbox_inches='tight', dpi=400)
    if kwargs['show']:
        plt.show()
    plt.close()


def generate_data(delta, xlabel, ylabel, zlabel, xlim_bahamas=(13.0, 14.7), xlim_macsis=(14.8, 15.5)):

    import numpy as np
    import pandas as pd

    x = []
    y = []
    z = []
    t = []
    fg = []
    xline = []

    for iz in ['z0p00', 'z0p25', 'z0p50', 'z1p00']:
        data = pd.read_csv('./data/BAHAMAS_%s.csv' % iz)
        #data = data[data[xlabel + delta + 'c_Msun'] > np.power(10, xlim_bahamas[0])]
        print "BAHAMAS %s: %i"%(iz, len(data))
        x += [np.log10(np.array(data[xlabel + delta + 'c_Msun']))]
        y += [np.log10(np.array(data[ylabel + delta + 'c_Msun']))]
        z += [np.log10(np.array(data[zlabel + delta + 'c_Msun']))]
        t += [np.log10(np.array(data[zlabel + delta + 'c_Msun']) + np.array(data[ylabel + delta + 'c_Msun']))]
        fg += [np.log10(np.array(data[ylabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun']))]

        xline += [np.linspace(xlim_bahamas[0], xlim_bahamas[1], 21)]

    """
    xx = []
    yy = []
    zz = []
    tt = []
    ffg = []
    for iz in ['z0p00', 'z0p50', 'z1p00']:
        data = pd.read_csv('./data/BAHAMAS_%s.csv' % iz)
        data = data[data[xlabel + delta + 'c_Msun'] > 7e13]
        data = data[data[xlabel + delta + 'c_Msun'] < 7e14]
        xx += list(np.log10(np.array(data[xlabel + delta + 'c_Msun'])))
        yy += list(np.log10(np.array(data[ylabel + delta + 'c_Msun'])))
        zz += list(np.log10(np.array(data[zlabel + delta + 'c_Msun'])))
        tt += list(np.log10(np.array(data[zlabel + delta + 'c_Msun']) + np.array(data[ylabel + delta + 'c_Msun'])))
        ffg += list(np.log10(np.array(data[ylabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun'])))

    x += [np.array(xx)]
    y += [np.array(yy)]
    z += [np.array(zz)]
    t += [np.array(tt)]
    fg += [np.array(ffg)]

    xline += [np.linspace(14.0, 15.0, 11)]
    """

    for iz in ['z0p00', 'z0p24', 'z0p46']:
        data = pd.read_csv('./data/MACSIS_%s.csv' % iz)
        #data = data[data[xlabel + delta + 'c_Msun'] > np.power(10, xlim_macsis[0])]
        print "MACSIS %s: %i"%(iz, len(data))
        x += [np.log10(np.array(data[xlabel + delta + 'c_Msun']))]
        y += [np.log10(np.array(data[ylabel + delta + 'c_Msun']))]
        z += [np.log10(np.array(data[zlabel + delta + 'c_Msun']))]
        t += [np.log10(np.array(data[zlabel + delta + 'c_Msun']) + np.array(data[ylabel + delta + 'c_Msun']))]
        fg += [np.log10(np.array(data[ylabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun']))]

        xline += [np.linspace(xlim_macsis[0], xlim_macsis[1], 5)]

    """
    xx = []
    yy = []
    zz = []
    tt = []
    ffg = []
    for iz in ['z0p00', 'z0p24', 'z0p46']:
        data = pd.read_csv('./data/MACSIS_%s.csv' % iz)
        #data = data[data[xlabel + delta + 'c_Msun'] > 6e14]
        xx += list(np.log10(np.array(data[xlabel + delta + 'c_Msun'])))
        yy += list(np.log10(np.array(data[ylabel + delta + 'c_Msun'])))
        zz += list(np.log10(np.array(data[zlabel + delta + 'c_Msun'])))
        tt += list(np.log10(np.array(data[zlabel + delta + 'c_Msun']) + np.array(data[ylabel + delta + 'c_Msun'])))
        ffg += list(np.log10(np.array(data[ylabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun'])))

    x += [np.array(xx)]
    y += [np.array(yy)]
    z += [np.array(zz)]
    t += [np.array(tt)]
    fg += [np.array(ffg)]
    xline += [np.linspace(14.5, 15.6, 5)]
    """
    return x, y, z, t, fg, xline


def master_generate_data(delta, xlabel, ylabel, zlabel):

    import numpy as np
    import pandas as pd

    x = []
    y = []
    z = []
    t = []
    fg = []
    fs = []
    fb = []
    xline = []

    for iz in ['z0p00', 'z0p25', 'z0p50']:
        data = pd.read_csv('./data/master_%s.csv' % iz)
        data = data[data[xlabel + delta + 'c_Msun'] > 1e13]
        data = data[(data[xlabel + delta + 'c_Msun'] < 1e14) +
                    (data[ylabel + delta + 'c_Msun'] * 1e14 > np.power(10, 12.5) * data[xlabel + delta + 'c_Msun']) ]
        x += [np.log10(np.array(data[xlabel + delta + 'c_Msun']))]
        y += [np.log10(np.array(data[ylabel + delta + 'c_Msun']))]
        z += [np.log10(np.array(data[zlabel + delta + 'c_Msun']))]
        t += [np.log10(np.array(data[zlabel + delta + 'c_Msun']) + np.array(data[ylabel + delta + 'c_Msun']))]
        fg += [np.log10(np.array(data[ylabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun']))]
        fs += [np.log10(np.array(data[zlabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun']))]
        fb += [np.log10(np.array(data[ylabel + delta + 'c_Msun']) / np.array(data[xlabel + delta + 'c_Msun']))]

        xline += [np.linspace(13.0, 15.2, 31)]

    return x, y, z, t, fg, xline


def test_evrard_et_al_2014():

    from number_count_modules import number_count_class
    nc = number_count_class()

    xlabel = 'M'
    ylabel = 'Mgas'
    zlabel = 'Mstar'
    tlabel = 'Mb'
    Msun = r'$[{\rm M}_{\odot}]$'
    filter_type = 'gaussian'
    show = False
    delta = '500'

    xp = 14.0
    deg = 3

    labels = [r'Z = 0', r'Z = 0.25', r'Z = 0.5', r'Z = 1']
    colors = ["#08420D", "#257F27", "#67E568", "#FFB62B"]

    x, y, z, t, fg, xline = generate_data(delta, xlabel, ylabel, zlabel)


    n = nc.number_count_vector(x, xline, labels)
    coef = nc.fit_polynomial(n, xline, xp=xp, deg=deg)
    for i in range(4): print coef[i]
    nexp = nc.expected_number_count(xline, coef, xp)
    nc.plot_actual_vs_fit(xline, n, labels=labels, colors=colors, xp=xp, deg=deg)
    plt.savefig(fdir_plot+'MF.png', bbox_inches='tight', dpi=400)

    # mf_slope = nc.mass_function_local_slope(xline, coef, xp=xp)
    # nc.predict_expected_number_count_obs(x, x, n, xline, ylim=(13.0, 13.85), nybins=20, xcut=14.0,
    #                                     labels=labels, colors=colors, xp=xp, deg=deg)
    # nc.predict_expected_number_count_obs(x, y, n, xline, ylim=(12.0, 13.5), nybins=20, xcut=14.0,
    #                                     labels=labels, colors=colors, xp=xp, deg=deg)
    nc.predict_expected_mass(x, y, coef, xline, ylim=(11.6, 13.5), nybins=30,
                             labels=labels, colors=colors, xp=xp,
                             xlabel= r'$\log_{10}($ Mgas %s $)$'%Msun, ylabel=r'$\log_{10}($ M$_{500}$ %s $)$'%Msun)
    plt.savefig(fdir_plot+'E14-Mgas', bbox_inches='tight', dpi=400)
    nc.predict_expected_mass(x, z, coef, xline, ylim=(12.0, 13.2), nybins=30, loc=4,
                             labels=labels, colors=colors, xp=xp,
                             xlabel= r'$\log_{10}($ Mstar %s $)$'%Msun, ylabel=r'$\log_{10}($ M$_{500}$ %s $)$'%Msun)
    plt.savefig(fdir_plot+'E14-Mstar.png', bbox_inches='tight', dpi=400)
    pass



def mass_function_constrain():

    from number_count_modules import number_count_class, number_count_likelihood_class
    nc = number_count_class()
    nlike = number_count_likelihood_class()

    xlabel = 'M'
    ylabel = 'Mgas'
    zlabel = 'Mstar'
    tlabel = 'Mb'
    Msun = r'$[{\rm M}_{\odot}]$'
    filter_type = 'gaussian'
    show = False
    delta = '500'

    xp = 14.0
    yp = 13.0
    deg = 3

    labels = [r'Z = 0', r'Z = 0.25', r'Z = 0.5', r'Z = 1']
    colors = ["#08420D", "#257F27", "#67E568", "#FFB62B"]
    alpha = [0.12, 0.12, 0.12, 0.12]
    fit_model = ['Local', 'Local', 'Local', 'Local']

    x, y, z, t, fg, xline = generate_data(delta, xlabel, ylabel, zlabel)

    gas_line = [np.linspace(12.0, 13.7, 103)]*4

    # MOR plot (Mass calibration)
    """
    plot_caller(plot_regression, [y, x], gas_line, white=True,
                var_num=2, labels=labels, fname='master_regression_%s_%s_' % (xlabel, ylabel),
                xlabel=ylabel, ylabel=xlabel, xunit=Msun, yunit=Msun, alpha=alpha,
                delta=delta, filter_type=filter_type, fit_model=fit_model, nrow=1,
                legend_loc=4, legend_fontsize=12, legend_title=None,
                ylim=(13, 15.0), xlim=(11.5, 14.0), show=show)
    plot_caller(plot_mass_evolution, [y, x], gas_line,  white=True,
                var_num=2, labels=labels[:], fname='master_mass_evolution_%s_%s_' % (xlabel, ylabel),
                xlabel=ylabel, ylabel=xlabel, xunit=Msun, yunit=Msun, alpha=alpha,
                delta=delta, filter_type=filter_type, fit_model=fit_model, nrow=2,
                legend_loc=1, legend_fontsize=14,
                xlim=(11.5, 14.0), ylim=[(0.5, 1.0), (0, 0.2)], show=show)

    n = nc.number_count_vector(x, xline, labels)
    coef = nc.fit_polynomial(n, xline, xp=xp, deg=deg)
    for i in range(4): print coef[i]
    nexp = nc.expected_number_count(xline, coef, xp)

    # Mass Function Plot
    nc.plot_actual_vs_fit(xline, n, labels=labels, colors=colors, xp=xp, deg=deg)
    plt.savefig(fdir_plot+'MF.png', bbox_inches='tight', dpi=400)
    """

    # nc.predict_number_count_obs(x, xline, y, gas_line, labels, colors, xp=xp, yp=yp, deg=deg)
    nlike.run_number_count_model(x[0], xline[0], y[0], gas_line[0], xp=xp, yp=yp, deg=deg)


    # mf_slope = nc.mass_function_local_slope(xline, coef, xp=xp)
    # nc.predict_expected_number_count_obs(x, x, n, xline, ylim=(13.0, 13.85), nybins=20, xcut=14.0,
    #                                     labels=labels, colors=colors, xp=xp, deg=deg)
    # nc.predict_expected_number_count_obs(x, y, n, xline, ylim=(12.0, 13.5), nybins=20, xcut=14.0,
    #                                     labels=labels, colors=colors, xp=xp, deg=deg)
    # nc.predict_expected_mass(x, y, coef, xline, ylim=(11.6, 13.5), nybins=30,
    #                          labels=labels, colors=colors, xp=xp,
    #                          xlabel= r'$\log_{10}($ Mgas %s $)$'%Msun, ylabel=r'$\log_{10}($ M$_{500}$ %s $)$'%Msun)
    # plt.savefig(fdir_plot+'E14-Mgas', bbox_inches='tight', dpi=400)
    # nc.predict_expected_mass(x, z, coef, xline, ylim=(12.0, 13.2), nybins=30, loc=4,
    #                          labels=labels, colors=colors, xp=xp,
    #                          xlabel= r'$\log_{10}($ Mstar %s $)$'%Msun, ylabel=r'$\log_{10}($ M$_{500}$ %s $)$'%Msun)
    # plt.savefig(fdir_plot+'E14-Mstar.png', bbox_inches='tight', dpi=400)
    pass


def rm_mass_function_constrain():

    from number_count_modules_obs import number_count_likelihood_class
    nlike = number_count_likelihood_class()
    # nlike.run_number_count_model(x[0], xline[0], y[0], gas_line[0], xp=xp, yp=yp, deg=deg)

