
import numpy as np
import matplotlib.pylab as plt
from scr.covariance_package import covariance_package
from astropy.cosmology import Planck13 as cosmo
import pymc
import seaborn as sns

fdir_plot = './plots/'

fdir_plot = './plots/'

class number_count_likelihood_class():

    def __init__(self):

        self.cov = covariance_package()
        self.cosmo = cosmo

    def calc_volume(self, z_index, z_bin, h=0.7, sky_frac=1.0):

        def calc_dV(zbin):
            dV = self.cosmo.comoving_volume(zbin[1]) - self.cosmo.comoving_volume(zbin[0])
            dV = sky_frac * dV.value * h ** 3
            return dV * 1e-6

        # dV = [calc_dV([z_bin[z_index[i]], z_bin[z_index[i]+1]]) for i in range(len(z_index))] (SLOW APPROACH)
        dV = calc_dV([z_bin[z_index], z_bin[z_index+1]])

        return dV

    def calc_dobs(self, obs_index, obs_bin):

        return obs_bin[1] - obs_bin[0]

    def fit_polynomial(self, n, xline, xp=14.0, deg=3):

        coef = []

        for i in range(len(n)):

            x = xline[i]
            x = (x[1:] + x[:-1]) / 2.0

            coef += [np.polyfit(x-xp, np.log(n[i]), deg=deg)]

        return coef

    def _fit_polynomial(self, n, xline, xp=14.0, deg=3):
        x = (xline[1:] + xline[:-1]) / 2.0
        coef = np.polyfit(x-xp, np.log(n), deg=deg)
        return coef

    def expected_number_count(self, xline, coef, xp=14.0):

        y = []

        for i in range(len(coef)):
            p = np.poly1d(coef[i])
            y += [np.exp(p(xline[i]-xp))]

        return y

    def __mass_function_local(self, x, coef, xp=14.0):

        deg = len(coef)-1

        p = np.poly1d(coef)
        A = np.exp(p(x - xp))

        c = coef
        c = [c[j]*(deg-j) for j in range(deg)]
        p = np.poly1d(c)
        s = p(x-xp)

        return A, s

    def mass_function_local_slope(self, xline, coef, xp=14.0):

        s = []
        deg = len(coef[0])-1

        for i in range(len(coef)):
            c = coef[i]
            c = [c[j]*(deg-j) for j in range(deg)]
            p = np.poly1d(c)
            s += [p(xline[i]-xp)]

        return s

    def _set_legend(self):

        legend = plt.legend(loc=1, fontsize=12, fancybox=True, shadow=True)
        frame = legend.get_frame()
        legend.set_frame_on(True)
        frame.set_facecolor('white')
        plt.setp(legend.get_title(), fontsize=12)

    def _set_ticksize(self, size=16, minor_size=8):

        plt.tick_params(axis='both', which='major', labelsize=size)
        plt.tick_params(axis='both', which='minor', labelsize=minor_size)

    def likelihood_model1(self, nObs, yObs, Slope, Norm, Sig, dx, xp=14, deg=3):

        # (1) Calculate the expected Mass -> MCR
        # (2) Calculate the slope and scatter parameter in -> MOR
        # (3) Calculate Number Count
        # (4) Write the likelihood

        mu = Slope * yObs + Norm
        # alpha = 1.0 / Slope # First Order Approximation
        # sigma = Sig / Slope # First Order Approximation

        # [beta_n, beta_n-1, beta_n-2, ...]
        beta = [pymc.Normal('beta_%i'%i, mu=0., tau=0.0001, value=0.0, observed=False) for i in range(deg+1)]

        @pymc.deterministic(plot=False)
        def exp_n(beta=beta, mu=mu, deg=deg, slope=Slope, sig=Sig, dx=dx):
            # It returns the normalization and the first order approximation for the scatter
            # MF = A x exp(beta1 x mu)

            p = np.poly1d(beta)
            A = dx * np.exp(p(mu))

            c = [beta[j] * (deg - j) for j in range(deg)]
            p = np.poly1d(c)
            beta1 = p(mu)

            return A * slope * np.exp(- sig**2 * beta1)

        likelihood = pymc.Poisson('n_obs', mu=exp_n, value=nObs, observed=True)

        return locals()

    def likelihood_model2(self, nObs, yObs, Slope, Norm, Sig, dx, xp=14, deg=3):

        # (1) Calculate the expected Mass -> MCR
        # (2) Calculate the slope and scatter parameter in -> MOR
        # (3) Calculate Number Count
        # (4) Write the likelihood

        # alpha = 1.0 / Slope # First Order Approximation
        # sigma = Sig / Slope # First Order Approximation
        slope = Slope #pymc.Normal('slope', mu=Slope, tau=100.0, value=Slope, observed=False)
        norm = pymc.Normal('norm', mu=Norm, tau=100.0, value=Norm, observed=False)
        sig = pymc.Normal('sig', mu=Sig, tau=100.0, value=Sig, observed=False)

        # [beta_n, beta_n-1, beta_n-2, ...]
        beta = [pymc.Normal('beta_%i'%i, mu=0., tau=0.0001, value=0.0, observed=False) for i in range(deg+1)]

        @pymc.deterministic(plot=False)
        def mu(yObs=yObs, slope=slope, norm=norm):
            return slope * yObs + norm

        @pymc.deterministic(plot=False)
        def exp_n(beta=beta, mu=mu, deg=deg, slope=slope, sig=sig, dx=dx):
            # It returns the normalization and the first order approximation for the scatter
            # MF = A x exp(beta1 x mu)

            p = np.poly1d(beta)
            A = dx * np.exp(p(mu))

            c = [beta[j] * (deg - j) for j in range(deg)]
            p = np.poly1d(c)
            beta1 = p(mu)

            return A * slope * np.exp(- sig**2 * beta1)

        likelihood = pymc.Poisson('n_obs', mu=exp_n, value=nObs, observed=True)

        return locals()

    def likelihood_model3(self, nObs, yObs, Slope, Norm, Sig, dx, xp=14, deg=3):

        # (1) Calculate the expected Mass -> MCR
        # (2) Calculate the slope and scatter parameter in -> MOR
        # (3) Calculate Number Count
        # (4) Write the likelihood

        # alpha = 1.0 / Slope # First Order Approximation
        # sigma = Sig / Slope # First Order Approximation
        print np.mean(Slope)
        print Slope
        print Norm
        print Slope * yObs + Norm
        print nObs
        print (Sig[10] - Sig[4]) / (Slope[10]*yObs[10] + Norm[10] - Slope[4]*yObs[4] - Norm[4])
        # exit()

        slope = pymc.Normal('slope', mu=0.7, tau=100.0, value=0.7, observed=False)
        slope_mu = pymc.Normal('slope_mu', mu=0.0, tau=100.0, value=0.0, observed=False)
        norm = pymc.Normal('norm', mu=-9.0, tau=100.0, value=-9.0, observed=False)
        sig = 0.15 #pymc.Uniform('sig', 0.001, 0.4, value=np.mean(Sig), observed=False)

        # [beta_n, beta_n-1, beta_n-2, ...]
        beta = [pymc.Normal('beta_%i'%i, mu=0., tau=0.0001, value=0.0, observed=False) for i in range(deg+1)]

        @pymc.deterministic(plot=False)
        def mu(yObs=yObs, slope=slope, norm=norm, slope_mu=slope_mu):
            return (slope + slope_mu*yObs) * yObs + norm

        @pymc.deterministic(plot=False)
        def exp_n(beta=beta, mu=mu, deg=deg, slope=slope, sig=sig, dx=dx):
            # It returns the normalization and the first order approximation for the scatter
            # MF = A x exp(beta1 x mu)

            p = np.poly1d(beta)
            A = dx * np.exp(p(mu))

            c = [beta[j] * (deg - j) for j in range(deg)]
            p = np.poly1d(c)
            beta1 = p(mu)

            return A * slope * np.exp(- np.array(sig)**2 * beta1)

        likelihood = pymc.Poisson('n_obs', mu=exp_n, value=nObs, observed=True)

        return locals()

    def likelihood_model_obs(self, nObs, yObs, dObs, dV, xp=14, deg=3):

        slope = pymc.Normal('slope', mu=1.0, tau=1600.0, value=1.0, observed=False)
        slope_mu = 0.0 #pymc.Normal('slope_mu', mu=0.0, tau=100.0, value=0.0, observed=False)
        norm = 0.0 #pymc.Normal('norm', mu=-9.0, tau=100.0, value=-9.0, observed=False)
        sig = 0.0 #pymc.Normal('slope', mu=1.0, tau=1600.0, value=1.0, observed=False)

        # [beta_n, beta_n-1, beta_n-2, ...]
        beta = [pymc.Normal('beta_%i'%i, mu=0., tau=0.0001, value=0.0, observed=False) for i in range(deg+1)]

        @pymc.deterministic(plot=False)
        def mu(yObs=yObs, slope=slope, norm=norm, slope_mu=slope_mu):
            return (slope + slope_mu*yObs) * yObs + norm

        @pymc.deterministic(plot=False)
        def exp_n(beta=beta, mu=mu, deg=deg, slope=slope, sig=sig, dx=dObs, dV=dV):
            # It returns the normalization and the first order approximation for the scatter
            # MF = A x exp(beta1 x mu)

            p = np.poly1d(beta)
            A = dx * np.exp(p(mu))

            c = [beta[j] * (deg - j) for j in range(deg)]
            p = np.poly1d(c)
            beta1 = p(mu)

            return dV * A * slope * np.exp(- np.array(sig)**2 * beta1)

        likelihood = pymc.Poisson('n_obs', mu=exp_n, value=nObs, observed=True)

        return locals()

    def likelihood_model_obs2(self, nObs, yObs, dObs, dV, expMu, slope_pri, sig_pri, deg=3):

        slope = slope_pri #pymc.Normal('slope', mu=1.0, tau=1600.0, value=1.0, observed=False)
        sig = sig_pri #pymc.Normal('slope', mu=1.0, tau=1600.0, value=1.0, observed=False)

        # [beta_n, beta_n-1, beta_n-2, ...]
        beta = [pymc.Normal('beta_%i'%i, mu=0., tau=0.0001, value=0.0, observed=False) for i in range(deg+1)]

        @pymc.deterministic(plot=False)
        def exp_n(beta=beta, mu=expMu, deg=deg, slope=slope, sig=sig, dx=dObs, dV=dV):
            # It returns the normalization and the first order approximation for the scatter
            # MF = A x exp(beta1 x mu)

            p = np.poly1d(beta)
            A = dx * np.exp(p(mu))

            c = [beta[j] * (deg - j) for j in range(deg)]
            p = np.poly1d(c)
            beta1 = p(mu)

            return dV * A * slope * np.exp(np.array(sig)**2 * beta1)

        likelihood = pymc.Poisson('n_obs', mu=exp_n, value=nObs, observed=True)

        return locals()

    def scaling_parameters(self, x, y, GaussianWidth=1.0):

        cov = covariance_package()

        xline = np.linspace(np.min(x), np.max(x), 21)
        xline = (xline[1:] + xline[:-1]) / 2.0

        slope = np.zeros(len(xline))
        norm = np.zeros(len(xline))
        sig = np.zeros(len(xline))

        for i in range(len(xline)):
            w = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=GaussianWidth)
            norm[i], slope[i], sig[i] = cov.linear_regression(x, y, weight=w)

        return slope, sig, norm

    def scaling_parameters_at(self, x, y, xe=1.0, GaussianWidth=1.0):

        cov = covariance_package()

        w = cov.calculate_weigth(x, weigth_type='gussian', mu=xe, sig=GaussianWidth)
        norm, slope, sig = cov.linear_regression(x, y, weight=w)

        return slope, sig, norm

    def print_results(self, M, deg=3, model_id=0, prefix='plot'):

        from lib import corner

        betas_m = [np.mean(M.trace('beta_%i'%i)[::]) for i in range(deg + 1)]
        betas_std = [np.std(M.trace('beta_%i'%i)[::]) for i in range(deg + 1)]

        print " Expected betas : "
        print betas_m
        print betas_std

        betas_post = np.array([M.trace('beta_%i' % i)[::] for i in range(deg + 1)]).T

        plt.clf()
        corner(betas_post,
               labels=[r"$\beta_3$",r"$\beta_2$",r"$\beta_1$",r"$\beta_0$"],
               quantiles=[0.16, 0.5, 0.84], label_kwargs={"fontsize": 14},
               show_titles=True, title_kwargs={"fontsize": 15})
        plt.savefig('./plots/%s-post-like-%i.png'%(prefix, model_id), bbox_inches='tight')
        plt.close()

    def print_inf_MF(self, M, nObs, yObs, dObs, dV, xp, n_dz=4, deg=3, model_id=0, prefix='plot'):

        ln10 = np.log(10)
        zmin = 0.1; dz=0.05

        betas_post = np.array([M.trace('beta_%i'%i)[::] for i in range(deg + 1)]).T
        post = np.array([np.exp( np.poly1d(betas_post[i])(yObs)) for i in range(len(betas_post))])
        post_med = np.percentile(post, 50, axis=0)
        post_up = np.percentile(post, 50+95/2., axis=0)
        post_down = np.percentile(post, 50-95/2., axis=0)

        plt.clf()
        sns.set_style("white")
        c = ['pink', 'salmon', 'orangered', 'brown', 'darkred']

        fig1 = plt.figure(1)

        # Plot Data-model
        frame1 = fig1.add_axes((.1, .5, .8, .6))

        ndens = nObs/dV/dObs

        for i in range(n_dz):
            plt.errorbar(yObs[i::n_dz]/ln10+xp, ndens[i::n_dz],
                         yerr=np.sqrt(ndens[i::n_dz]), fmt='o', color=c[i],
                         label=r'$z \in [%0.2f-%0.2f]$'%(i*dz+zmin,(i+1)*dz+zmin))
        plt.fill_between(yObs/ln10+xp, post_down, post_up, facecolor='green', alpha=0.5)
        plt.plot(yObs/ln10+xp, post_med, '-', c='green', lw=1.2)

        betas = [-0.05, -0.30, -1.81, 1.975]
        true = np.exp(np.poly1d(betas)(yObs))
        plt.plot(yObs/ln10+xp, true, '--', c='k', lw=1.75)


        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        frame1.set_ylabel(r'${\rm d} n/ {\rm d} \log (M)$ [ Mpc$^{-3}$ $h^{-3}$  ]', size=18)
        frame1.set_yscale('log')
        frame1.set_xlim([13, 14.8])
        frame1.set_ylim([1e-2, 1e3])
        legend = frame1.legend(loc=3, fontsize=15, fancybox=True, shadow=True)
        frame = legend.get_frame()
        legend.set_frame_on(True)

        plt.grid(True)

        frame2 = fig1.add_axes((.1, .1, .8, .4))

        # plt.errorbar(yObs+xp, ndens/post_med - 1.0, yerr=np.sqrt(ndens)/post_med, fmt='o')
        for i in range(n_dz):
            plt.errorbar(yObs[i::n_dz]/ln10+xp, ndens[i::n_dz]/true[i::n_dz] - 1.0,
                         yerr=np.sqrt(ndens[i::n_dz])/true[i::n_dz], fmt='o', color=c[i])
        plt.fill_between(yObs/ln10+xp, post_down/true - 1.0, post_up/true - 1.0,
                         facecolor='green', alpha=0.3)
        plt.plot(yObs/ln10+xp, post_med/true-1.0, '--', c='g', lw=1.25)
        plt.plot([13, 14.8], [0, 0], '-', c='k', lw=2.5)

        # plt.fill_between([13, 13.5], [-1, -1], [1, 1], facecolor='darkgrey', alpha=0.3)

        frame2.set_xlabel(r'$\log(M_{500c}  [{\rm M}_{\odot}])$', size=20)
        frame2.set_ylim([-0.4, 0.4])
        frame2.set_xlim([13, 14.8])
        frame2.set_yticklabels([-0.4, ' ', -0.2, ' ', 0, ' ', 0.2, ' ', ' '])
        frame2.set_ylabel(r'$\delta N / N_{\rm fit} - 1$', size=18)

        plt.grid(True)

        plt.savefig('./plots/%s-post-fit-MF-%i.png'%(prefix, model_id), bbox_inches='tight')
        plt.close()
        # plt.show()

    def print_fit(self, M, nObs, yObs, dObs, dV, expMu, slope, sig, yp, n_dz=4, deg=3,
                  model_id=0, prefix='plot', xlim=[20, 200], xlabel=r'$S_{rm obs}$'):

        def exp_n(beta, mu=expMu, deg=deg, slope=slope, sig=sig, dx=dObs, dV=dV):
            p = np.poly1d(beta)
            A = dx * np.exp(p(mu))

            c = [beta[j] * (deg - j) for j in range(deg)]
            p = np.poly1d(c)
            beta1 = p(mu)

            return dV * A * slope * np.exp(np.array(sig) ** 2 * beta1)

        ln10 = np.log(10.0)
        zmin = 0.1; dz=0.05

        betas_post = np.array([M.trace('beta_%i'%i)[::] for i in range(deg + 1)]).T
        post = np.array([ exp_n(betas_post[i]) for i in range(len(betas_post))])
        post_med = np.percentile(post, 50, axis=0)
        post_up = np.percentile(post, 50+95/2., axis=0)
        post_down = np.percentile(post, 50-95/2., axis=0)

        ndens = nObs

        c = ['pink', 'salmon', 'orangered', 'brown', 'darkred']

        plt.clf()
        sns.set_style("white")
        fig1 = plt.figure(1)

        # Plot Data-model
        frame1 = fig1.add_axes((.1, .5, .8, .6))

        for i in range(n_dz):
            plt.errorbar(np.exp(yObs[i::n_dz])*yp, ndens[i::n_dz], yerr=np.sqrt(ndens[i::n_dz]), fmt='o', color=c[i],
                         label=r'$z \in [%0.2f-%0.2f]$' % (i*dz+zmin, (i+1)*dz+zmin))
            plt.fill_between(np.exp(yObs[i::n_dz])*yp, post_down[i::n_dz], post_up[i::n_dz], facecolor=c[i], alpha=0.5)
            plt.plot(np.exp(yObs[i::n_dz])*yp, post_med[i::n_dz], '-', c=c[i], lw=1.5)

        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        frame1.set_ylabel(r'Number Count [ $1 / d \log (M)$  ]', size=18)
        frame1.set_yscale('log')
        frame1.set_xscale('log', subsx=[2, 4, 6,  8])
        frame1.set_xlim(xlim)
        legend = frame1.legend(loc=3, fontsize=15, fancybox=True, shadow=True)
        frame = legend.get_frame()
        legend.set_frame_on(True)

        plt.grid(True, which='major', axis='both', color='darkgrey', lw=1.5, linestyle='-')
        plt.grid(True, which='minor', axis='x', color='darkgrey', lw=0.5, linestyle='-')

        frame2 = fig1.add_axes((.1, .1, .8, .4))

        # plt.errorbar(yObs+xp, ndens/post_med - 1.0, yerr=np.sqrt(ndens)/post_med, fmt='o')
        for i in range(n_dz):
            plt.errorbar(np.exp(yObs[i::n_dz])*yp, ndens[i::n_dz]/post_med[i::n_dz] - 1.0,
                         yerr=np.sqrt(ndens[i::n_dz])/post_med[i::n_dz], fmt='o', color=c[i])
            plt.fill_between(np.exp(yObs[i::n_dz])*yp, post_down[i::n_dz]/post_med[i::n_dz] - 1.0,
                             post_up[i::n_dz]/post_med[i::n_dz] - 1.0, facecolor=c[i], alpha=0.2)
        plt.plot(xlim, [0, 0], '-', c='k', lw=1.5)

        # plt.fill_between([13, 13.5], [-1, -1], [1, 1], facecolor='darkgrey', alpha=0.3)

        frame2.set_xlabel(xlabel, size=20)
        frame2.set_ylim([-0.4, 0.4])
        frame2.set_xscale('log', subsx=[2, 4, 6,  8])
        frame2.set_xlim(xlim)
        frame2.set_yticklabels([-0.4, ' ', -0.2, ' ', 0, ' ', 0.2, ' ', ' '])
        frame2.set_ylabel(r'$\delta N / N_{\rm fit} - 1$', size=18)

        import matplotlib.ticker as ticker
        frame2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        frame2.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())

        plt.grid(True, which='major', axis='both', color='darkgrey', lw=1.5, linestyle='-')
        plt.grid(True, which='minor', axis='x', color='darkgrey', lw=0.5, linestyle='-')

        plt.savefig('./plots/%s-post-fit-Data-%i.png'%(prefix, model_id), bbox_inches='tight')
        plt.close()
        # plt.show()

    def run_number_count_model(self, x, xline, y, yline, xp=14, yp=13, deg=2):

        cov = covariance_package()

        nObs, dx = self.number_count_vector(y, yline)
        print("Number of Halos : %i"%sum(nObs))
        yObs = (yline[1:] + yline[:-1]) / 2.0

        slope = np.zeros(len(yObs))
        norm = np.zeros(len(yObs))
        sig = np.zeros(len(yObs))

        for i in range(len(yObs)):
            w = cov.calculate_weigth(y, weigth_type='gussian', mu=yObs[i], sig=GaussianWidth)
            norm[i], slope[i], sig[i] = cov.linear_regression(y, (x - xp), weight=w)

        # Beild the model
        model = self.likelihood_model1(nObs, yObs, slope, norm, sig, dx, xp=14, deg=deg)

        # RUN MCMC
        # do the MCMC sampling 10000 times and save all steps after 1000 initial steps
        M = pymc.MCMC(model)

        M.sample(iter=40000, burn=10000, verbose=0)

        # save posterior distribution
        self.print_results(M, x, xline, xp, deg=3, model_id=5)
        self.print_inf_MF(M, x, xline, xp, deg=deg, model_id=5)
        self.print_fit(M, x, xline, xp, deg=deg, model_id=5)

        # print "Slope %0.2f +- %0.2f"%(np.mean(M.trace('slope')[::]), np.std(M.trace('slope')[::]))
        # print "Sig %0.2f +- %0.2f"%(np.mean(M.trace('sig')[::]), np.std(M.trace('sig')[::]))
        # print "Norm %0.2f +- %0.2f"%(np.mean(M.trace('norm')[::]), np.std(M.trace('norm')[::]))

        # self.save_posteriors(M)


