import numpy as np

class lightcone():

    def __init__(self):
        pass


    def LC_HMF_method2(self, haloMassFunction, zrange, zbins=10,\
                             sky_frac=1.0, log_scale=True):
        """ return lightcone halo mass function
               input: 
                    halo mass function : hmf objec (MassFunction)
                    zrang : redshif range of interest (0.1-0.2)
                    zbin : number of redshift bins (for integration)
                    sky_frac : fraction of aky covered by the survey
                    log_scale : Return dN / dlogM or dN / dM
               output:
                    light cone halo mass function
        """ 

        z = np.linspace(zrange[0],zrange[1],zbins+1); dz=z[1]-z[0]
        z = ( z[1:] + z[:-1] ) / 2.
        
        redshift_integrated_hmf = np.zeros(len(haloMassFunction.dndlnm))

        for iZ in z: 

            # update redshift of halo mass function
            haloMassFunction.update(z=iZ)

            # get dV / dz / dSteradian
            dVdz = haloMassFunction.cosmo.differential_comoving_volume(iZ)

            # get dV / dz of the survey area
            dVdz = 4. * np.pi * sky_frac * dVdz

            if log_scale:
                redshift_integrated_hmf += haloMassFunction.dndlog10m * dVdz * dz
            else:
                redshift_integrated_hmf += haloMassFunction.dndm * dVdz * dz

        #return   redshift_integrated_hmf 
        return ( redshift_integrated_hmf[1:] + redshift_integrated_hmf[:-1] ) / 2. 
       
        
    def LC_HMF(self, haloMassFunction, zrange, zbins=10, sky_frac=1.0, log_scale=True):
        """ return lightcone halo mass function
               input: 
                    halo mass function : hmf objec (MassFunction)
                    zrang : redshif range of interest (0.1-0.2)
                    zbin : number of redshift bins (for integration)
                    sky_frac : fraction of aky covered by the survey
                    log_scale : Return dN / dlogM or dN / dM
               output:
                    light cone halo mass function
        """ 

        z = np.linspace(zrange[0],zrange[1],zbins+1); dz=z[1]-z[0]
        
        redshift_integrated_hmf = np.zeros(len(haloMassFunction.dndlnm))

        h = haloMassFunction.cosmo.h

        for iZmin, iZmax in zip(z[:-1],z[1:]): 

            # update redshift of halo mass function
            haloMassFunction.update(z=(iZmin+iZmax)/2.)

            # get dV [for all sky]
            dV = haloMassFunction.cosmo.comoving_volume(iZmax) -\
                 haloMassFunction.cosmo.comoving_volume(iZmin)

            # get dV of the survey area
            dV = sky_frac * dV * h**3

            if log_scale:
                redshift_integrated_hmf += haloMassFunction.dndlog10m * dV
            else:
                redshift_integrated_hmf += haloMassFunction.dndm * dV

        #return   redshift_integrated_hmf 
        return ( redshift_integrated_hmf[1:] + redshift_integrated_hmf[:-1] ) / 2. 
       



    def LC_HMF_approx(self, haloMassFunction, zrange,\
                            sky_frac=1.0, log_scale=True):
        """ return approximate lightcone halo mass function
            NOTE: this function is not very useful though
                  it is fast but not accurate at all
               input: 
                    halo mass function : hmf objec (MassFunction)
                    zrang : redshif range of interest (0.1-0.2)
                    sky_frac : fraction of aky covered by the survey
                    log_scale : Return dN / dlogM or dN / dM
               output:
                    light cone halo mass function
        """ 
 
        zMean = ( zrange[0] + zrange[1] ) /2.

        # mass function at the mean redshift
        haloMassFunction.update(z=zMean)

        # total volume of survey
        V = haloMassFunction.cosmo.comoving_volume(zrange[1]) -\
            haloMassFunction.cosmo.comoving_volume(zrange[0])
        V = V * sky_frac
        
        if log_scale:
            return V*(haloMassFunction.dndlog10m[1:]+haloMassFunction.dndlog10m[:-1])/2.
        else:
            return V*(haloMassFunction.dndm[1:]+haloMassFunction.dndm[:-1])/2.

        


""" Halo Mass Function Proxy 
this is a convinient tool to play with mass function 
the math follows Evrard et al. (2014) """
class hmf_proxy:
   
    def __init__(self, hmf):
        self.A_coef = []
        self.beta_coef = []
        self.approx = False  # whether approximation exists
        self.m_p = 1e14
        self.hmf = hmf
        pass


    def power_low_approx_fixed_redshift(self, m_p=1e14, order=2):
        """ This is approximation for mass function based on Evrard et al. (2014)
            convention. It returns the normalization and an array of n^th order 
            approximation for the mass funcion at given redshift

               dn / dmu = A exp( - beta1 mu - beta2 mu^2 - ... )

               input: 
                    halo mass function : hmf objec (MassFunction)
                    m_p : pivot mass
                    order : number of Taylor expantions
               output:
                    normalization + array of Taylor expantion              
        """ 
 
        # fit a power low to mass function
        lnM = np.log(self.hmf.m) - np.log(m_p)
        lndndlnM = np.log(self.hmf.dndlnm)

        coefs = np.polyfit(lnM, lndndlnM, order)
 
        return coefs[-1], -coefs[-2::-1]


    def power_low_approx(self, \
                               m_p=1e14, z_p=0.2, z_order=3, m_order=4,\
                               mrange=(13.0,15.5), zrange=(0.0,1.0), zbins=10,\
                               **kwargs):
        """ This function approximate mass function based on Evrard et al. (2014)
            convention. It returns the normalization and an array of n^th order
            coefficients for the mass funcion.

               dn / dmu = A(z) exp( - beta1(z) mu - beta2(z) mu^2 - ... )
               where mu = ln(M / M_p) & z = z-z_p
               A(z) = A_0 + A_1 x z + A_2 x z^2 + ... (same for betas)
               input:
                    halo mass function : hmf objec (MassFunction)
                    m_p(z_p) : pivot mass (redshift)
                    mrange(zrange) : mass range (redshift range)
                    m(z)_order : number of polynomials
                    z_bin : number of redshift bins
                    **kwargs : to update hmf
               output:
                    normalization + array of Taylor expantion
        """
        z = np.linspace(zrange[0], zrange[1], zbins+1)
        z = ( z[1:] + z[:-1] ) / 2.
        self.hmf.update(Mmin=mrange[0], Mmax=mrange[1], **kwargs)

        # save value of A and betas for each redshift
        A = []
        beta = []

        for iZ in z: 

            self.hmf.update(z=iZ)
            Ai, betai = self.power_low_approx_fixed_redshift(m_p=m_p, order=m_order)
            A    += [Ai]
            beta += [betai]

        beta = np.array(beta)

        # redshift regression for A and betas
        self.A_coef = np.polyfit(z, A, z_order)[-1::-1]

        self.beta_coef = []
        for iB in beta.T:
            self.beta_coef += [np.polyfit(z, iB, z_order)[-1::-1]] 

        self.beta_coef = np.array(self.beta_coef).T

        self.approx = True
        self.m_p = m_p



    def return_polynomial_coefs(self, z):
        """ return the approximation for polynomial coefs at fixed redshift """

        if not self.approx:
            raise PolyProxError(' Error: first fit a function')

        zorder = len(self.A_coef)
        A = sum(self.A_coef[i]*np.power(z,i) for i in range(zorder)) 
        beta = sum(self.beta_coef[i]*np.power(z,i) for i in range(zorder))

        return A, beta


    def return_slope(self, M, z):
        """ return the slope pf mass function at given redshift and mass
               input:
                    M : mass
                    z : redshift
               output:
                    return beta_1(M,z)

        """

        if not self.approx:
            raise PolyProxError(' Error: first fit a function')

        _, beta = self.return_polynomial_coefs(z)
    
        mu = np.log(M/self.m_p)
        
        return sum(float(i+1)*beta[i]*np.power(mu,i) for i in range(len(beta)))



    def return_mean_scale(self, z, new_lnm_p):
        """ return the mean and scale of mass function
                  dn/dlogn \propto exp(-(mu - mean)^2 / 2 scale)
               input:
                    M : mass
                    z : redshift
               output:
                    return mu, scale
        """

        _, beta =  self.return_polynomial_coefs(z)

        mean  = - beta[0] / beta[1] / 2. - np.log(self.m_p) + new_lnm_p 
        scale = 1. / beta[1]   

        return mean, scale




    def return_approx_mass_function(self):
        """ return approximate mass function based on the polynomial fit
            for given redshift. The redshift and masses are implied from 
            hmf object.
               input:
                    haloMassFunction : hmf objec (MassFunction)
               output:
                    return mass function dndlnm
        """

        if not self.approx:
            raise PolyProxError(' Error: first fit a function')

        A, beta = self.return_polynomial_coefs(self.hmf.z)
     
        mu = np.log(self.hmf.m/self.m_p)
        
        return np.exp(A) *\
               np.exp(-sum(beta[i]*np.power(mu,i+1) for i in range(len(beta))))



    def log_likelihood_hmf(self, mass=1e14, z=-1, fixed_redshift=True):
        """ The function approximate mass function with n^th order
            polynomial function based on Evrard et al. (2014)
            convention. It returns the log-likelihood function 
            for entire range of redshift or fixed redshift
            [the first one is not implimented yet]

            [Note that it is not nomalized properly]

               dn / dmu = A exp( - beta1 mu - beta2 mu^2 - ... )

               input: 
                    mass : mass of halo
                    z : redshift
                        if z == -1 then it assumes redshift of hmf object
               output:
                    return log-likelihood
        """ 

        if not self.approx:
            raise PolyProxError(' Error: first fit a function')

        if not fixed_redshift: 
            raise NotImplementedError(" Wait for future upgrades.")

        mu = np.log(mass / self.m_p)

        if z == -1:
             A, beta = self.return_polynomial_coefs(self.hmf.z)
        else:
             A, beta = self.return_polynomial_coefs(z)

        return -sum(beta[i]*np.power(mu,i+1) for i in range(len(beta)))
 
      
        

""" TEST GOES WONDERFUL :) """

"""
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(-1,'/home/aryaf/Pipelines/libraries/cosm/')

import numpy.random as npr

from matplotlib.pylab import plt
from hmf import MassFunction
from hmf import Cosmology
from astropy.cosmology import Planck13, FLRW, WMAP5, WMAP7, WMAP9#, Planck15

hmfT = MassFunction(Mmin=13., Mmax=16, delta_h=500.0,\
                    cosmo_model=WMAP9, delta_wrt='crit')

def Ez(z):
    return np.sqrt(0.7 + 0.3*(1.0+z)**3)


lc = lightcone()
m = hmfT.m
dm = m[1:] - m[:-1]

#6.42, the slope is 1.49, and the intrinsic scatter is ~50%.
#The pivot point is 7.34 h_{70}^{-1}10^14Msun

m     = ( m[1:] + m[:-1] ) / 2. / (7.34e14 * 0.7)
# m     = ( m[1:] + m[:-1] ) / 2. / (7.34e14)
slopeD = 1.32 # RASS
normD  = 3.43 # RASS
scattD = 0.66 # RASS

# slopeD = 1.32 # NEW
# normD  = 8.42 # NEW
# scattD = 0.39 # NEW

LxLim1= 7.0
LxLim2= 4.4

zrange1 = (0.15, 0.24)
zrange2 = (0.24, 0.30)

dNdm  = lc.LC_HMF(hmfT, zrange1, sky_frac=0.38, zbins=100, log_scale=False)
dN1 = dNdm * dm
dNdm  = lc.LC_HMF(hmfT, zrange2, sky_frac=0.38, zbins=100, log_scale=False)
dN2 = dNdm * dm

Ntot = []
clusters = []
for i in range(100):
    #print i
    counter = 0
    realization_cluster = []

    slope = slopeD + npr.normal(0.0, 0.47)
    norm  = normD + npr.normal(0.0, 0.4)
    scatt = scattD + npr.normal(0.0, 0.17)

    for zrange, dN, LxLim in zip([zrange1, zrange2],\
                                 [dN1, dN2], [LxLim1, LxLim2]):


        for idndm, im in zip(dN, m):

            N = npr.poisson(idndm)
            if N == 0: continue

            lnm = npr.normal(np.log(im), 0.2, size=N)
            evol = Ez((zrange[0]+zrange[1])/2.0)

            # generate random cluster
            lnLxBar = np.log(norm) + slope * (lnm + np.log(evol))
            lnLx    = npr.normal(lnLxBar, scatt)
            Lx      = np.exp(lnLx) 

            # count clusters above threshold 
            mask = Lx > LxLim
            try: len(mask)
            except:
               if mask:
                   counter += 1
                   realization_cluster += [ Lx ]
               continue

            counter += sum(mask)
            realization_cluster += list(Lx[mask])

    print counter, len(realization_cluster)

    values, base = np.histogram(np.log10(realization_cluster), bins=100, range=(0.0,2.0))

    #evaluate the cumulative
    cumulative = np.cumsum(values); value = np.log10(counter-cumulative)
    plt.plot(base[:-1], value, c='magenta', alpha=0.1)

    Ntot += [counter]

import pandas as pd
data = pd.read_csv('../obsdata/Sarah_r/Lxrass.dat', delimiter=r"\s+")

def ez(z):
    return np.sqrt(0.7 + 0.3*(1.0+z)**3)

z = np.array(data['z'])
logLx = np.log10(data['Lxrass(10^44)'] / ez(z))

values, base = np.histogram(logLx, bins=100, range=(0.0,2.0))

#evaluate the cumulative
cumulative = np.cumsum(values)
value = np.log10(len(data)-cumulative)
plt.plot(base[:-1], value, c='black', linewidth=2, label='LoCuSS')

plt.title('WMAP9')
plt.xlabel('log(Lxrass) [x 10^44 ergs/s]', size=18)
plt.ylabel('log(Cumulative # Clusters)', size=18)
plt.xlim([0.,2.3])
plt.ylim([0,np.log10(500)])
plt.legend(loc=1)

plt.savefig('WMAP9-Cluster.png', bbox_inches='tight')
plt.show()
"""





