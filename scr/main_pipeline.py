#!/usr/bin/env python2.7

import numpy as np
import pymc


def rm_mass_function_constrain():
    from load_data import load_data, prep_data, prep_true, prep_obs_data
    from number_count_modules_obs import number_count_likelihood_class

    deg = 3
    xp = 14.0
    slope = 1.14
    sig = 0.40
    norm = -0.02

    fname1 = 'RM_training.csv'
    fname2 = 'XCAT_Aardvark_1.0.fit' #'aardvark-1.0-dr8_run_redmapper_v6.3.3_lgt20_catalog.fit'
    nlike = number_count_likelihood_class()

    print("Load Data : %s" % fname1)
    df1 = load_data(fname=fname1, csv=True, mp=1, op=30.0, obs_label='LAMBDA', mass_label='M500')
    print("Load Data : %s" % fname2)
    df2 = load_data(fname=fname2, csv=False, mp=1e14, op=1, obs_label='HALOID', mass_label='M500')

    print("Estimate the Scaling Relation")
    slope_inf, sig_inf, norm_inf = nlike.scaling_parameters_at(
        np.array(df1['LAMBDA']), np.array(df1['M500']), xe=0.0, GaussianWidth=2.0)
    print slope_inf, norm_inf, sig_inf

    print("Filter Data and bin them")
    z_bin, obs_bin, z_index, obs_index, count, exp_obs, exp_mass, mean_mass = \
        prep_obs_data(df1, norm, slope, obs_label='LAMBDA', mass_label='M500', z_label='Z')
    true_z_bin, true_bin, true_z_index, true_index, true_count, true_mass = prep_true(df2, true_label='M500', z_label='Z')

    #    import matplotlib.pylab as plt
    #    plt.plot(mean_mass, exp_mass, 'r.')
    #    plt.show()

    print("Calculate the Volume of Each Bin")
    volume = nlike.calc_volume(z_index, z_bin, h=0.7, sky_frac=0.24)
    true_volume = nlike.calc_volume(true_z_index, true_z_bin, h=0.7, sky_frac=0.24)
    dObs = nlike.calc_dobs(obs_index, obs_bin)
    dmass = nlike.calc_dobs(true_index, true_bin)

    print("Total Number of Clusters : %i" % sum(count))

    # slope_inf, sig_inf = nlike.scaling_parameters(np.array(df1['LAMBDA']), np.array(df1['M500']))
    # print np.mean(slope_inf)
    # print np.mean(sig_inf)
    # exit()

    # Build the model
    model = nlike.likelihood_model_obs2(count, exp_obs, dObs, volume, exp_mass, slope, sig, deg=deg)

    print("Run MCMC")
    # RUN MCMC
    M = pymc.MCMC(model)
    M.sample(iter=40000, burn=10000, verbose=-1)

    # save posterior distribution
    nlike.print_results(M, deg=deg, model_id=5, prefix='RM-Aardvark-')
    nlike.print_inf_MF(M, true_count, true_mass, dmass, true_volume, xp, n_dz=4, deg=deg, model_id=5,
                       prefix='RM-Aardvark-')
    nlike.print_fit(M, count, exp_obs, dObs, volume, exp_mass, slope, sig, 30.0, n_dz=4, deg=deg, model_id=5,
                    prefix='RM-Aardvark-', xlim=[20, 200], xlabel=r'$\lambda_{\rm RM}$')

    # print "Slope %0.2f +- %0.2f"%(np.mean(M.trace('slope')[::]), np.std(M.trace('slope')[::]))
    # print "Sig %0.2f +- %0.2f"%(np.mean(M.trace('sig')[::]), np.std(M.trace('sig')[::]))
    # print "Norm %0.2f +- %0.2f"%(np.mean(M.trace('norm')[::]), np.std(M.trace('norm')[::]))

    # self.save_posteriors(M)


def mass_function_constrain():

    from load_data import load_data, prep_data, prep_true
    from number_count_modules_obs import number_count_likelihood_class

    deg = 3
    xp = 14.0
    slope = 1.0
    sig = 0.05

    fname = 'XCAT_Aardvark_1.0.fit'
    nlike = number_count_likelihood_class()

    print("Load Data : %s"%fname)
    df = load_data(fname=fname)

    print("Filter Data and bin them")
    z_bin, obs_bin, z_index, obs_index, count, exp_obs, exp_mass = \
        prep_data(df, slope, sig, obs_label='Mass_obs', mass_label='M500', z_label='Z')
    true_z_bin, true_bin, true_z_index, true_index, true_count, true_mass = prep_true(df, true_label='M500', z_label='Z')

    print("Calculate the Volume of Each Bin")
    volume = nlike.calc_volume(z_index, z_bin, h=0.7, sky_frac=0.24)
    true_volume = nlike.calc_volume(true_z_index, true_z_bin, h=0.7, sky_frac=0.24)
    dObs = nlike.calc_dobs(obs_index, obs_bin)
    dmass = nlike.calc_dobs(true_index, true_bin)

    print("Total Number of Clusters : %i" % sum(count))

    # Build the model
    model = nlike.likelihood_model_obs2(count, exp_obs, dObs, volume, exp_mass, slope, sig, deg=deg)

    print("Run MCMC")
    # RUN MCMC
    M = pymc.MCMC(model)
    M.sample(iter=40000, burn=10000, verbose=0)

    # save posterior distribution
    nlike.print_results(M, xp, deg=deg, model_id=5, prefix='RM-%ip-'%(100*sig))
    nlike.print_inf_MF(M, true_count, true_mass, dmass, true_volume, xp, n_dz=4, deg=deg, model_id=5, prefix='RM-%ip-'%(100*sig))
    nlike.print_fit(M, count, exp_obs, dObs, volume, exp_mass, slope, sig, xp, n_dz=4, deg=deg, model_id=5, prefix='RM-%ip-'%(100*sig))

    # print "Slope %0.2f +- %0.2f"%(np.mean(M.trace('slope')[::]), np.std(M.trace('slope')[::]))
    # print "Sig %0.2f +- %0.2f"%(np.mean(M.trace('sig')[::]), np.std(M.trace('sig')[::]))
    # print "Norm %0.2f +- %0.2f"%(np.mean(M.trace('norm')[::]), np.std(M.trace('norm')[::]))

    # self.save_posteriors(M)

