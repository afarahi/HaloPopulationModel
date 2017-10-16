#!/usr/bin/env python2.7

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def load_data(fname, fdir='./data/', labels=None, csv=False,
              mp=1e14, op=30.0, obs_label='Mass_obs', mass_label='M500'):

    if csv:
        df = pd.read_csv(fdir+fname)
    else:
        if labels is None:
            dat = Table.read(fdir+fname, format='fits')
        else:
            dat = Table.read(fdir+fname, format='fits')[labels]
        df = dat.to_pandas()
        del dat

    df[mass_label] = np.log(df[mass_label] / mp)
    df[obs_label] = np.log(df[obs_label] / op)

    return df

def prep_obs_data(df, norm_inf, slope_inf, obs_label='Mass_obs', mass_label='M500', z_label='Z'):

    def bin_index(val, bin_min, bin_max, bin_num, log=False):
        if log:
            index = bin_num * (np.log(val) - np.log(bin_min)) / (np.log(bin_max) - np.log(bin_min))
        else:
            index = bin_num * (val - bin_min) / (bin_max - bin_min)
        return int(index)

    # Filter out data
    Z_max = 0.3
    Z_min = 0.1

    obs_max = np.log(200.0/30.0)
    obs_min = np.log(20.0/30.0)

    df = df[df[z_label] < Z_max]
    df = df[df[z_label] > Z_min]
    df = df[df[obs_label] < obs_max]
    df = df[df[obs_label] > obs_min]

    # Binning Data
    Z_bins = 4
    Obs_bins = 4

    df['Obs_bins_id'] = np.zeros(len(df))

    df['Z_bins_id'] = df[z_label].apply(bin_index, args=(Z_min, Z_max, Z_bins, False))
    df['Obs_bins_id'] = df[obs_label].apply(bin_index, args=(obs_min, obs_max, Obs_bins, False))


    count = np.array(df.groupby(['Obs_bins_id', 'Z_bins_id'])[obs_label].count())
    exp_obs = np.array(df.groupby(['Obs_bins_id', 'Z_bins_id'])[obs_label].mean())
    mean_mass = np.array(df.groupby(['Obs_bins_id', 'Z_bins_id'])[mass_label].mean())
    exp_mass = slope_inf * exp_obs + norm_inf

    z_bin = np.linspace(Z_min, Z_max, Z_bins+1)
    obs_bin = np.linspace(obs_min, obs_max, Obs_bins+1)

    obs_index = [i for i in range(Obs_bins)]
    obs_index = np.repeat(obs_index, Z_bins)

    z_index = np.array([i for i in range(Z_bins)] * Obs_bins)

    return z_bin, obs_bin, z_index, obs_index, count, exp_obs, exp_mass, mean_mass


def prep_data(df, slope, sig, obs_label='Mass_obs', mass_label='M500', z_label='Z'):

    def bin_index(val, bin_min, bin_max, bin_num, log=False):
        if log:
            index = bin_num * (np.log(val) - np.log(bin_min)) / (np.log(bin_max) - np.log(bin_min))
        else:
            index = bin_num * (val - bin_min) / (bin_max - bin_min)
        return int(index)

    # Filter out data
    Z_max = 0.3
    Z_min = 0.1

    obs_max = np.log(7.0)
    obs_min = np.log(0.15)

    df[obs_label] = np.random.normal(df[mass_label]/slope, sig)

    df = df[df[z_label] < Z_max]
    df = df[df[z_label] > Z_min]
    df = df[df[obs_label] < obs_max]
    df = df[df[obs_label] > obs_min]

    # Binning Data
    Z_bins = 4
    Obs_bins = 10

    df['Obs_bins_id'] = np.zeros(len(df))

    df['Z_bins_id'] = df[z_label].apply(bin_index, args=(Z_min, Z_max, Z_bins, False))
    df['Obs_bins_id'] = df[obs_label].apply(bin_index, args=(obs_min, obs_max, Obs_bins, False))


    count = np.array(df.groupby(['Obs_bins_id', 'Z_bins_id'])[obs_label].count())
    exp_obs = np.array(df.groupby(['Obs_bins_id', 'Z_bins_id'])[obs_label].mean())
    exp_mass = np.array(df.groupby(['Obs_bins_id', 'Z_bins_id'])[mass_label].mean())

    z_bin = np.linspace(Z_min, Z_max, Z_bins+1)
    obs_bin = np.linspace(obs_min, obs_max, Obs_bins+1)

    obs_index = [i for i in range(Obs_bins)]
    obs_index = np.repeat(obs_index, Z_bins)

    z_index = np.array([i for i in range(Z_bins)] * Obs_bins)

    return z_bin, obs_bin, z_index, obs_index, count, exp_obs, exp_mass


def prep_true(df, true_label='M500', z_label='Z'):

    def bin_index(val, bin_min, bin_max, bin_num, log=False):
        if log:
            index = bin_num * (np.log(val) - np.log(bin_min)) / (np.log(bin_max) - np.log(bin_min))
        else:
            index = bin_num * (val - bin_min) / (bin_max - bin_min)
        return int(index)

    # Filter out data
    Z_max = 0.3
    Z_min = 0.1

    true_max = np.log(7.0)
    true_min = np.log(0.1)

    df = df[df[z_label] < Z_max]
    df = df[df[z_label] > Z_min]
    df = df[df[true_label] < true_max]
    df = df[df[true_label] > true_min]

    # Binning Data
    Z_bins = 4
    True_bins = 10

    df['True_bins_id'] = np.zeros(len(df))

    df['Z_bins_id'] = df[z_label].apply(bin_index, args=(Z_min, Z_max, Z_bins, False))
    df['True_bins_id'] = df[true_label].apply(bin_index, args=(true_min, true_max, True_bins, False))

    count = np.array(df.groupby(['True_bins_id', 'Z_bins_id'])[true_label].count())
    true_mass = np.array(df.groupby(['True_bins_id', 'Z_bins_id'])[true_label].mean())

    true_bin = np.linspace(true_min, true_max, True_bins+1)

    true_index = [i for i in range(True_bins)]
    true_index = np.repeat(true_index, Z_bins)

    true_z_bin = np.linspace(Z_min, Z_max, Z_bins+1)
    true_z_index = np.array([i for i in range(Z_bins)] * True_bins)

    return true_z_bin, true_bin, true_z_index, true_index, count, true_mass