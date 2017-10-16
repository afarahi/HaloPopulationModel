#!/usr/bin/env python2.7

import pandas as pd
import h5py as h5
import numpy as np

def join(data, label, val=None):
    if val == None: z = pd.DataFrame({label:np.zeros(len(data))})
    else: z = pd.DataFrame({label:val})
    return data.join(z)

labels = ['SO_masses_radii_true','Variables_0_r500c_true',\
          'BCG_properties_30kpc']

fdirLoad = '/Users/aryaf/Desktop/codes/PyCharms/covProject/data/hdf5/'
fdirSave = '/Users/aryaf/Desktop/codes/PyCharms/covProject/data/'

#fileName = 'BAHAMAS_vol2_snap32.hdf5'

id = 3
#for iz in [26]:#,28,30,32]:
for iz in ['z0p00','z0p24', 'z0p46', 'z1p00']:
    #fileName = 'BAHAMAS_vol%i_snap%i.hdf5'%(id,iz)
    #fileName = 'BAHAMAS_%s.hdf5'%iz
    fileName = 'MACSIS_%s.hdf5'%iz

    f = h5.File(fdirLoad+fileName, "r")
    data = pd.DataFrame( f['GroupNumber'][:] )
    #print f['SO_masses_radii_true'].values()
    #print f.values()
    #print f['Header'].keys()
    #exit()


    for ilabel in labels:
        sub_labels = f[ilabel].keys()
        print sub_labels
        for jlabel in sub_labels:
            data_to_join = f[ilabel][jlabel]
            print ilabel, jlabel,
            if sum(data_to_join) == 0.0: 
                print f[ilabel][jlabel][:]
                continue
            print ilabel, jlabel, min(f[ilabel][jlabel][:])
            data = join(data, jlabel, data_to_join)
    f.close()

    data = join(data, 'Rhill')

    #for i in range(len(data)):
    #    Rhill =
    #    data['Rhill'].iloc[i] = min(Rhill)


    data.to_csv(fdirSave+fileName[:-5]+'.csv', index=False)

