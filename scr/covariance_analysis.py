#!/usr/bin/env python2.7

from plot_modules import *

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


def covariance_pipeline():

    import numpy as np
    import pandas as pd

    #x, y, z = make_xyz(ndata=1600)

    # labels
    #M200c_Msun,M2500c_Msun,M500c_Msun,
    #Mgas200c_Msun,Mgas2500c_Msun,Mgas500c_Msun,
    #Mstar200c_Msun,Mstar2500c_Msun,Mstar500c_Msun,
    #r200c_Mpc,r2500c_Mpc,r500c_Mpc,
    #E_kinetic_thermal,Lx0p5_2p0_erg_s,Y_Da2_Mpc2,
    #kTmw_keV,kTspec_keV,SFR_Msu

    xlabel = 'M'; ylabel='Mgas'; zlabel='Mstar'; tlabel='Mb'
    filter_type = 'gaussian'

    for iz in [32, 30, 28, 26]:
        for delta in ['500']:

            x = []; y = []; z = []; t = []
            for id in [1, 2, 3]:
                 data = pd.read_csv('./data/BAHAMAS_vol%i_snap%i.csv'%(id, iz))
                 #data = data[data[xlabel+delta+'c_Msun']>1e14]
                 x += [np.log10(np.array(data[xlabel+delta+'c_Msun']))]
                 y += [np.log10(np.array(data[ylabel+delta+'c_Msun']))]
                 z += [np.log10(np.array(data[zlabel+delta+'c_Msun']))]
                 t += [np.log10(np.array(data[zlabel+delta+'c_Msun']) + np.array(data[ylabel+delta+'c_Msun']))]

            xline = np.linspace(13.0, 15.0, 41)
            if delta == '2500':
                xline = np.linspace(13.0, 14.5, 41)

            #xline = np.linspace(np.log(10**13.0), np.log(10**15.0), 41)
            #if delta == '2500':
            #    xline = np.linspace(np.log(10**13.0), np.log(10**14.5), 41)


            #plot_correlation(x, y, z, xline,\
            #                 xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, delta=delta,\
            #                 filter_type=filter_type, iz=iz, show=True)

            #plot_cumulative_correlation(x, y, z, xline,\
            #                            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, delta=delta,\
            #                            filter_type=filter_type, iz=iz, show=False)

            #plot_regression(x[2], y[2], xline,\
            #                xlabel=xlabel, ylabel=ylabel, delta=delta,\
            #                filter_type=filter_type, iz=iz, show=False)
            #plot_regression(x[2], z[2], xline,\
            #                xlabel=xlabel, ylabel=zlabel, delta=delta,\
            #                filter_type=filter_type, iz=iz, show=False)

            #plot_mass_evolution(x, y, xline,
            #                    xlabel=xlabel, delta=delta, title=ylabel,
            #                    filter_type=filter_type, iz=iz, show=False)
            #plot_mass_evolution(x, z, xline, title=zlabel,
            #                    xlabel=xlabel, delta=delta,
            #                    filter_type=filter_type, iz=iz, show=False)
            plot_mass_evolution(x, t, xline, title=tlabel,
                                xlabel=xlabel, delta=delta,
                                filter_type=filter_type, iz=iz, show=True)

            #dy = plot_residual(x, y, xline, bins=120, lim=(-1.2, 1.2),
            #                   xlabel=ylabel, delta=delta,
            #                   filter_type=filter_type, iz=iz, show=True)
            #dz = plot_residual(x, z, xline, bins=120, lim=(-0.6,0.6),\
            #    xlabel=zlabel, delta=delta,\
            #    filter_type=filter_type, iz=iz, show=True)
            #print iz, delta, np.corrcoef(dy,dz)[0,1]

            #plot_regression(x[2], np.power(10.,y[2]-x[2]), xline,\
            #    xlabel=xlabel, ylabel='f_gas', delta=delta,\
            #    filter_type=filter_type, iz=iz, show=False)
            #plot_regression(x[2], np.power(10.,z[2]-x[2]), xline,\
            #    xlabel=xlabel, ylabel='f_star', delta=delta,\
            #    filter_type=filter_type, iz=iz, show=False)







def test_simulation():
    import numpy.random as npr
    import numpy as np
    #npr.seed(800)

    ndata = 800
    x = np.linspace(-5.0, 5.0, ndata)
    ran1 = npr.normal(0.0, 0.5, ndata)
    ran2 = npr.normal(0.0, 0.1, ndata)
    y = x*1.2 - 0.4 + ran1 + ran2
    z = x*0.8 + 0.2 + ran2

    cov = covariance_package()
    w = cov.calculate_weigth(x, weigth_type='gussian', mu=3.0, sig = 0.5)

    #print cov.linear_regression(x, y, weight=None)
    #print cov.linear_regression(x, z, weight=None)
    cov.calc_correlation_fixed_x(x, y, z, weight=None)

    #print cov.linear_regression(x, y, weight=w)
    #print cov.linear_regression(x, z, weight=w)
    cov.calc_correlation_fixed_x(x, y, z, weight=w)        
        
"""
if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    print "Start ... "
    #test_simulation()
    covariance_pipeline()
    print "... end. "
"""