
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from scr.covariance_package import covariance_package
import pymc
import seaborn as sns

fdir_plot = './plots/'
nBootstrap = 100
GaussianWidth = 0.2

fdir_plot = './plots/'

class number_count_likelihood_class():

    def __init__(self):

        self.nBootstrap = 100
        self.GaussianWidth = 0.2
        self.cov = covariance_package()

    def number_count_vector(self, x, xline):

        xlinep = xline
        dx = xlinep[1] - xlinep[0]
        xp = x

        ncp = []
        for xmin, xmax in zip(xlinep[:-1], xlinep[1:]):
            ncp += [np.sum((xp>xmin) * (xp <= xmax))]

        return np.array(ncp), dx

    def number_density_count_vector(self, x, xline):

        xlinep = xline
        dx = xlinep[1] - xlinep[0]
        xp = x

        ncp = []
        for xmin, xmax in zip(xlinep[:-1], xlinep[1:]):
            ncp += [np.sum((xp>xmin) * (xp <= xmax))]

        return np.array(ncp) / dx, dx

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

    def print_results(self, M, x, xline, xp, deg=3, model_id=0):

        from corner import corner

        betas_m = [np.mean(M.trace('beta_%i'%i)[::]) for i in range(deg + 1)]
        betas_std = [np.std(M.trace('beta_%i'%i)[::]) for i in range(deg + 1)]

        n, dx = self.number_density_count_vector(x, xline)
        xx = xline - xp
        xx = (xx[1:] + xx[:-1]) / 2.0
        betas_r = np.polyfit(xx, np.log(n), deg=deg)

        print " Expected betas : "
        print betas_m
        print betas_std
        print " Real betas : "
        print betas_r

        betas_post = np.array([M.trace('beta_%i' % i)[::] for i in range(deg + 1)]).T

        plt.clf()
        corner(betas_post,
               labels=[r"$\beta_3$",r"$\beta_2$",r"$\beta_1$",r"$\beta_0$"],
               quantiles=[0.16, 0.5, 0.84], truths=betas_r, label_kwargs={"fontsize": 14},
               show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('./plots/post-like-%i.png'%model_id, bbox_inches='tight')
        plt.close()

    def print_inf_MF(self, M, x, xline, xp, deg=3, model_id=0):

        nData, dx = self.number_density_count_vector(x, xline)
        xObs = (xline[1:] + xline[:-1]) / 2.0

        betas_post = np.array([M.trace('beta_%i'%i)[::] for i in range(deg + 1)]).T
        post = np.array([np.exp( np.poly1d(betas_post[i])(xObs-xp)) for i in range(len(betas_post))])
        post_med = np.percentile(post, 50, axis=0)
        post_up = np.percentile(post, 50+95/2., axis=0)
        post_down = np.percentile(post, 50-95/2., axis=0)

        plt.clf()
        sns.set_style("white")

        fig1 = plt.figure(1)

        # Plot Data-model
        frame1 = fig1.add_axes((.1, .5, .8, .6))

        plt.errorbar(xObs, nData, yerr=np.sqrt(nData/dx), fmt='o')
        plt.fill_between(xObs, post_down, post_up, facecolor='red', alpha=0.5)
        plt.plot(xObs, post_med, '-', c='r', lw=1.5)

        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        frame1.set_ylabel(r'Number Count [ $1 / d \log (M)$  ]', size=18)
        frame1.set_yscale('log')
        frame1.set_xlim([13, 14.8])
        plt.grid(True)

        frame2 = fig1.add_axes((.1, .1, .8, .4))

        plt.errorbar(xObs, nData/post_med - 1.0, yerr=np.sqrt(nData/dx)/post_med, fmt='o')
        plt.fill_between(xObs, post_down/post_med - 1.0, post_up/post_med - 1.0, facecolor='red', alpha=0.5)
        plt.plot([13, 14.8], [0, 0], '-', c='k', lw=1.5)

        plt.fill_between([13, 13.5], [-1, -1], [1, 1], facecolor='darkgrey', alpha=0.3)

        frame2.set_xlabel(r'$\log(M_{500c}  [{\rm M}_{\odot}])$', size=20)
        frame2.set_ylim([-0.2, 0.2])
        frame2.set_xlim([13, 14.8])
        frame2.set_yticklabels([-0.2, ' ', -0.1, ' ', 0, ' ', 0.1, ' ', ' '])
        frame2.set_ylabel(r'$\delta N / N_{\rm fit} - 1$', size=18)

        plt.grid(True)

        plt.savefig('./plots/post-fit-MF-%i.png'%model_id, bbox_inches='tight')
        plt.close()
        # plt.show()

    def print_fit(self, M, x, xline, xp, deg=3, model_id=0):

        nData, dx = self.number_density_count_vector(x, xline)
        xObs = (xline[1:] + xline[:-1]) / 2.0

        betas_post = np.array([M.trace('beta_%i'%i)[::] for i in range(deg + 1)]).T
        post = np.array([np.exp( np.poly1d(betas_post[i])(xObs-xp)) for i in range(len(betas_post))])
        post_med = np.percentile(post, 50, axis=0)
        post_up = np.percentile(post, 50+95/2., axis=0)
        post_down = np.percentile(post, 50-95/2., axis=0)

        plt.clf()
        sns.set_style("white")

        fig1 = plt.figure(1)

        # Plot Data-model
        frame1 = fig1.add_axes((.1, .5, .8, .6))

        plt.errorbar(xObs, nData, yerr=np.sqrt(nData/dx), fmt='o')
        plt.fill_between(xObs, post_down, post_up, facecolor='red', alpha=0.5)
        plt.plot(xObs, post_med, '-', c='r', lw=1.5)

        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        frame1.set_ylabel(r'Number Count [ $1 / d \log (M)$  ]', size=18)
        frame1.set_yscale('log')
        frame1.set_xlim([13, 14.8])
        plt.grid(True)

        frame2 = fig1.add_axes((.1, .1, .8, .4))

        plt.errorbar(xObs, nData/post_med - 1.0, yerr=np.sqrt(nData/dx)/post_med, fmt='o')
        plt.fill_between(xObs, post_down/post_med - 1.0, post_up/post_med - 1.0, facecolor='red', alpha=0.5)
        plt.plot([13, 14.8], [0, 0], '-', c='k', lw=1.5)

        plt.fill_between([13, 13.5], [-1, -1], [1, 1], facecolor='darkgrey', alpha=0.3)

        frame2.set_xlabel(r'$\log(M_{500c}  [{\rm M}_{\odot}])$', size=20)
        frame2.set_ylim([-0.2, 0.2])
        frame2.set_xlim([13, 14.8])
        frame2.set_yticklabels([-0.2, ' ', -0.1, ' ', 0, ' ', 0.1, ' ', ' '])
        frame2.set_ylabel(r'$\delta N / N_{\rm fit} - 1$', size=18)

        plt.grid(True)

        plt.savefig('./plots/post-fit-MF-%i.png'%model_id, bbox_inches='tight')
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



"""
    def hierarchical_block_model_cov(self, sl_pub, sl_pri, classpred_pub, classpred_pri, home_id, area_id, \
                                     sl_pub_obs, sl_pri_obs, covariance):

        num_homes = len(home_id)
        ones = np.ones(self.n_areas)
        dig = covariance

        # Priors
        mu_pri = pymc.Normal('mu_pri', mu=0., tau=0.0001, value=0.0, observed=False)
        nu_pri = pymc.Uniform('nu_pri', 0.0001, 100.0, value=1.0, observed=False)

        mu_pub = pymc.Normal('mu_pub', mu=0., tau=0.0001, value=0.0, observed=False)
        nu_pub = pymc.Uniform('nu_pub', 0.0001, 100.0, value=1.0, observed=False)

        beta_pri = pymc.Normal('beta_pri', mu=0., tau=0.0001, value=0.0, observed=False)
        beta_pub = pymc.Normal('beta_pub', mu=0., tau=0.0001, value=0.0, observed=False)

        scale_cov_pri = pymc.Uniform('scale_cov_pri', 0.0, 1.0, value=0.5, observed=False)
        scale_cov_pub = pymc.Uniform('scale_cov_pub', 0.0, 1.0, value=0.5, observed=False)

        @pymc.deterministic(plot=False)
        def cov_pub(r=scale_cov_pub, C=covariance, nu=nu_pub):
            cov = C * r
            np.fill_diagonal(cov, 1.0)
            return nu * cov

        @pymc.deterministic(plot=False)
        def cov_pri(r=scale_cov_pri, C=covariance, nu=nu_pri):
            cov = C * r
            np.fill_diagonal(cov, 1.0)
            return nu * cov

        alpha_pri = pymc.MvNormalCov('alpha_pri', mu=ones * mu_pri, C=cov_pri, value=ones * 0.0, observed=False)
        alpha_pub = pymc.MvNormalCov('alpha_pub', mu=ones * mu_pub, C=cov_pub, value=ones * 0.0, observed=False)

        @pymc.deterministic(plot=False)
        def exp_hidden_true_pub_z(alpha=alpha_pub, beta=beta_pub, classpred=classpred_pub, area_id=area_id):
            return np.array(alpha)[area_id] + beta * classpred

        @pymc.deterministic(plot=False)
        def exp_hidden_true_pri_z(alpha=alpha_pri, beta=beta_pri, classpred=classpred_pri, area_id=area_id):
            return np.array(alpha)[area_id] + beta * classpred

        @pymc.deterministic(plot=False)
        def hidden_prob_pub(hidden_true_pub_z=exp_hidden_true_pub_z):
            p = norm.cdf(hidden_true_pub_z)
            p[p < 0.01] = 0.01
            p[p > 0.99] = 0.99
            return p

        @pymc.deterministic(plot=False)
        def hidden_prob_pri(hidden_true_pri_z=exp_hidden_true_pri_z):
            p = norm.cdf(hidden_true_pri_z)
            p[p < 0.01] = 0.01
            p[p > 0.99] = 0.99
            return p

        obs_pub = pymc.Bernoulli('obs_pub', p=hidden_prob_pub[sl_pub_obs], value=sl_pub[sl_pub_obs], observed=True)
        obs_pri = pymc.Bernoulli('obs_pri', p=hidden_prob_pri[sl_pri_obs], value=sl_pri[sl_pri_obs], observed=True)

        return locals()

    def save_posteriors(self, M):

        nnn = 1000
        d = {'beta_public': M.trace('beta_pub')[nnn::],
             'beta_private': M.trace('beta_pri')[nnn::],
             'mu_alpha_public': M.trace('mu_pub')[nnn::],
             'mu_alpha_private': M.trace('mu_pri')[nnn::],
             'sig_alpha_public': np.sqrt(M.trace('nu_pub')[nnn::]),
             'sig_alpha_private': np.sqrt(M.trace('nu_pri')[nnn::]),
             'scale_cov_pub': M.trace('scale_cov_pub')[nnn::],
             'scale_cov_pri': M.trace('scale_cov_pri')[nnn::]
             }

        for i in range(self.n_areas):
            d.update({'alpha_private_%i' % i: (M.trace('alpha_pri')[nnn::].T)[i]})
            d.update({'alpha_public_%i' % i: (M.trace('alpha_pub')[nnn::].T)[i]})

        df_post = pd.DataFrame(data=d)
        df_post.to_csv(self.fdir + 'Posterior_MCMC_Cov_Model.csv')
        del d, df_post

    def run_mcmc_model(self):

        # Generate Numpy array to be used as an input of out Hierarchical Model

        # mask out all SL which both public and private SL are censored (FALSE)
        mask = self.df['sl_pri'].notnull() + self.df['sl_pub'].notnull()
        n_homes = sum(mask)

        home_id = np.array(self.df['PID no Dash'][mask])
        area_id = np.array(self.df[self.block_label][mask])

        sl_pri = np.array(self.df['sl_pri'][mask])
        sl_pub = np.array(self.df['sl_pub'][mask])

        sl_pri_obs = np.array(self.df['sl_pri_obs'][mask])
        sl_pub_obs = np.array(self.df['sl_pub_obs'][mask])

        classpred_pub = np.array(self.df['predicted_public_sl'][mask])
        classpred_pri = np.array(self.df['predicted_private_sl'][mask])

        print "Number of homes : ", n_homes
        print "Number of blocks : ", self.n_areas

        # Building the covariance matrix
        covariance = self.build_covariance_matrix(self.df)

        # RUN MCMC
        model = self.hierarchical_block_model_cov(sl_pub, sl_pri, classpred_pub, classpred_pri,
                                             home_id, area_id, sl_pub_obs, sl_pri_obs, covariance)

        M = pymc.MCMC(model)

        # do the MCMC sampling 1000 times and save all steps after 100 initial steps
        M.sample(iter=200000, burn=50000, verbose=0)

        # save posterior distribution
        self.save_posteriors(M)
"""

class number_count_class():

    def __init__(self):

        self.nBootstrap = 100
        self.GaussianWidth = 0.2
        self.cov = covariance_package()

    def number_count_vector(self, x, xline, labels):

        number_count = []

        for i in range(len(labels)):

            xlinep = xline[i]
            dx = xlinep[1] - xlinep[0]
            xp = x[i]

            ncp = []

            for xmin, xmax in zip(xlinep[:-1], xlinep[1:]):
                ncp += [np.sum((xp>xmin) * (xp <= xmax))]

            number_count += [np.array(ncp) / dx]

        return number_count

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

    """
    def __predict_expected_number_count_obs_method1(self, x, y, yline):

        nobs = []
        xexp = []
        dy = yline[1] - yline[0]
        for ymin, ymax in zip(yline[:-1], yline[1:]):
            mask = (y > ymin) * (y <= ymax)
            nobs += [np.sum(mask)]
            xexp += [np.mean(x[mask])]

        return np.array(nobs)/dy, np.array(xexp)

    def __predict_expected_number_count_obs_method2(self, x, y, n, xline, xexp, xp, deg):

        coef = self._fit_polynomial(n, xline, xp=xp, deg=deg)

        A, beta = self.__mass_function_local(xexp, coef, xp=14.0)

        slope = []
        sigma = []

        for i in range(len(xexp)):
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xexp[i], sig=self.GaussianWidth)
            intercept, slopel, sigl = self.cov.linear_regression(x, y, weight=w)
            slope += [slopel]
            sigma += [sigl]

        sigma = np.array(sigma)
        slope = np.array(slope)

        return A / slope * np.exp(- sigma**2 * beta / slope**2)

    def __predict_expected_number_count_obs_method3(self, x, y, n, xline, xexp, xp, deg):

        coef = self._fit_polynomial(n, xline, xp=xp, deg=deg)

        A, beta = self.__mass_function_local(xexp, coef, xp=14.0)

        slope = []
        sigma = []

        for i in range(len(xexp)):
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xexp[i], sig=self.GaussianWidth)
            intercept, slopel, sigl = self.cov.linear_regression(x, y, weight=w)
            slope += [slopel]
            sigma += [sigl]

        sigma = np.array(sigma)
        slope = np.array(slope)

        return A / slope * np.exp(- sigma**2 * beta / slope**2)

    def predict_expected_number_count_obs(self, x, y, number, xline, ylim, xcut=14.0, nybins=10,
                                          labels=None, colors=None, xp=14.0, deg=3):

        for i in range(len(number)):

            yline = np.linspace(ylim[0], ylim[1], nybins+1)

            nobs, xexp = self.__predict_expected_number_count_obs_method1(x[i], y[i], yline)
            nglob = self.__predict_expected_number_count_obs_method2(x[i], y[i], number[i], xline[i], xexp, xp, deg)

            yline = (yline[1:] + yline[:-1]) / 2.0
            plt.semilogy(yline, nobs, 'o', c=colors[i])
            plt.semilogy(yline, nglob, '-', c=colors[i], label=labels[i])

        plt.xlabel(r'$\log_{10}(M_{\rm gas}  [M_{\odot}])$', size=18)
        plt.ylabel(r'Number Count', size=15)

        # self._set_legend()
        self._set_ticksize(size=14)
        #plt.savefig(fdir_plot + 'number_count.pdf', bbox_inches='tight', dpi=400)
        plt.show()
    """

    def __predict_expected_mass_method1(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            w = self.cov.calculate_weigth(y, weigth_type='gussian', mu=yline[j], sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(y, x, weight=w)
            xline_exp[j] = slope * yline[j] + intercept

        return xline_exp

    def __predict_expected_mass_method2(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            intercept, slope, sig = self.cov.linear_regression(x, y)
            xline_exp[j] = (yline[j] - intercept) / slope

        return xline_exp

    def __predict_expected_mass_method3(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            xp = np.median(x[(y<yline[j]+0.1)*(y>yline[j]-0.1)])
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xp, sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(x, y, weight=w)
            xline_exp[j] = (yline[j] - intercept) / slope

        return xline_exp

    def __predict_expected_mass_method2(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            intercept, slope, sig = self.cov.linear_regression(x, y)
            xline_exp[j] = (yline[j] - intercept) / slope

        return xline_exp

    def __predict_expected_mass_method4(self, x, y, yline, coef, xp):

        xline_exp = np.zeros(len(yline))

        deg = len(coef)-1
        c = coef
        c = [c[j] * (deg - j) for j in range(deg)]
        p = np.poly1d(c)

        for j in range(len(xline_exp)):
            xl = np.median(x[(y<yline[j]+0.1)*(y>yline[j]-0.1)])
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xl, sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(x, y, weight=w)
            xline_exp[j] = (yline[j] - intercept) / slope

            beta_1 = p(xl - xp)
            xline_exp[j] += beta_1 * sig**2 / slope**2

        return xline_exp

    def __predict_expected_mass_method5(self, x, y, yline, coef, xp):

        xline_exp = np.zeros(len(yline))

        deg = len(coef)-1
        c = coef
        c = [c[j] * (deg - j) for j in range(deg)]
        p1 = np.poly1d(c)

        deg = len(c)-1
        c = [c[j] * (deg - j) for j in range(deg)]
        p2 = np.poly1d(c)

        for j in range(len(xline_exp)):
            xl = np.median(x[(y<yline[j]+0.1)*(y>yline[j]-0.1)])
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xl, sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(x, y, weight=w)
            xline_exp[j] = (yline[j] - intercept) / slope

            beta_1 = p1(xl - xp)
            beta_2 = p2(xl - xp)
            xline_exp[j] += beta_1 * sig**2 / slope**2 / (1.0 - beta_2 * sig**2 / slope**2)

        return xline_exp

    def predict_expected_mass(self, x, y, coef, xline, ylim, nybins=10, loc=2,
                              labels=None, colors=None, xp=14.0, xlabel=None, ylabel=None):

        #c = ["#E56124", "#E53E30", "#7F2353", "#F911FF"]
        c = ["#33cc33", "#ff3300", "#0000ff", "#a55927"]

        i = 0

        yline = np.linspace(ylim[0], ylim[1], nybins+1)

        xline_exp1 = self.__predict_expected_mass_method1(x[i], y[i], yline)
        xline_exp2 = self.__predict_expected_mass_method2(x[i], y[i], yline)
        xline_exp3 = self.__predict_expected_mass_method3(x[i], y[i], yline)
        xline_exp4 = self.__predict_expected_mass_method4(x[i], y[i], yline, coef[i], xp)
        xline_exp5 = self.__predict_expected_mass_method5(x[i], y[i], yline, coef[i], xp)

        sns.set_style("white")
        plt.grid(True)

        fig1 = plt.figure(1)

        plt.clf()

        # Plot Data-model
        frame1 = fig1.add_axes((.1, .5, .8, .6))

        # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        mask = x[i] < 13.6
        plt.scatter(y[i][mask], x[i][mask], s=0.5, color=colors[i], alpha=0.08)
        plt.scatter(y[i][~mask], x[i][~mask], s=0.5, color=colors[i], alpha=0.18)
        plt.plot(yline, xline_exp1, color=colors[i], linestyle='-', linewidth=1.3, label='Fit')
        plt.plot(yline, xline_exp2, color=c[0], linestyle='-', linewidth=1.3, label='Power-law prediction')
        plt.plot(yline, xline_exp3, color=c[1], linestyle='-', linewidth=1.3, label='Local power-law prediction')
        plt.plot(yline, xline_exp4, color=c[2], linestyle='-', linewidth=1.3, label='Local power-law prediction with MF correction')
        # plt.plot(yline, xline_exp5, color=c[2], linestyle='-', linewidth=1.3, label='Local power-law prediction with MF correction')

        legend = plt.legend(loc=loc, fontsize=12, fancybox=True, shadow=True)

        frame = legend.get_frame()
        legend.set_frame_on(True)
        frame.set_facecolor('white')

        frame1.set_ylim(13, 15)
        frame1.set_xlim(11.5, 13.7)

        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        frame1.set_yticklabels([' ', 13.5, 14, 14.5, 15])  # Remove x-tic labels for the first frame
        frame1.set_ylabel(ylabel, size=18)
        plt.grid()

        # Residual plot
        frame2 = fig1.add_axes((.1, .1, .8, .4))
        plt.plot([11,15], [0, 0], color='k', linestyle='-', linewidth=1.6)
        plt.plot([11,15], [0.01, 0.01], color='k', linestyle='--', linewidth=0.8)
        plt.plot([11,15], [-0.01, -0.01], color='k', linestyle='--', linewidth=0.8)
        plt.plot(yline, np.log(10.0)*(xline_exp2 - xline_exp1), color=c[0], linestyle='-', linewidth=2.2)
        plt.plot(yline, np.log(10.0)*(xline_exp3 - xline_exp1), color=c[1], linestyle='-', linewidth=2.2)
        plt.plot(yline, np.log(10.0)*(xline_exp4 - xline_exp1), color=c[2], linestyle='-', linewidth=2.2)
        # plt.plot(yline, (xline_exp5 - xline_exp1), color=c[3], linestyle='-', linewidth=2.2)

        #frame2.set_yticklabels([-0.1, -0.05, ' ', 0.0, ' ', 0.05])  # Remove x-tic labels for the first frame
        frame2.set_xlabel(xlabel, size=18)
        # frame2.set_ylabel(r'$\delta (\log M)$')
        frame2.set_ylabel(r'$\%$ Bias', size=18)

        frame2.set_ylim(-0.10, 0.10)
        frame2.set_xlim(11.5, 13.7)

        plt.grid()
        # plt.show()
        plt.savefig('')

    def _set_legend(self):

        legend = plt.legend(loc=1, fontsize=12, fancybox=True, shadow=True)
        frame = legend.get_frame()
        legend.set_frame_on(True)
        frame.set_facecolor('white')
        plt.setp(legend.get_title(), fontsize=12)

    def _set_ticksize(self, size=16, minor_size=8):

        plt.tick_params(axis='both', which='major', labelsize=size)
        plt.tick_params(axis='both', which='minor', labelsize=minor_size)

    def plot_actual_vs_fit(self, xline, n, labels=None, colors=None, xp=14.0, deg=3):

        sns.set_style("white")
        plt.grid(True)

        coef = self.fit_polynomial(n, xline, xp=xp, deg=deg)
        nexp = self.expected_number_count(xline, coef, xp=xp)

        ilab = ['0.0', '0.25', '0.5', '1.0']

        for i in range(len(n)):
            if i == 1: continue
            x = (xline[i][1:] + xline[i][:-1]) / 2.0
            plt.semilogy(x, n[i], 'o', c=colors[i])
            plt.semilogy(xline[i], nexp[i], '-', c=colors[i], label=labels[i])

            fname = './data/mVector_PLANCK-SMT-%s.txt'%ilab[i]
            x, y = np.loadtxt(fname, comments='#', delimiter=' ', usecols=(0,7), unpack=True)
            plt.plot(np.log10(x/0.6704), y*400*400*400, '--', c=colors[i], lw=1)


        plt.xlabel(r'$\log(M_{500c}  [{\rm M}_{\odot}])$', size=20)
        plt.ylabel(r'Number Count [ $1 / d \log (M)$  ]', size=18)
        plt.xlim([13, 15])
        plt.ylim([10, 100000])

        self._set_legend()
        self._set_ticksize(size=16)
        # plt.savefig(fdir_plot + 'number_count.pdf', bbox_inches='tight', dpi=400)

    """
    def __predict_expected_number_count_obs_method1(self, x, xline, y, yline, labels, xp=14, yp=13, deg=2):

        cov = covariance_package()
        n_pred = []

        n = self.number_count_vector(x, xline, labels)
        coef = self.fit_polynomial(n, xline, xp=xp, deg=deg)

        for j in range(4):

            slope = np.zeros(len(yline[j]))
            norm = np.zeros(len(yline[j]))
            sig = np.zeros(len(yline[j]))

            beta_2 = - 2.0 * coef[j][0]
            beta_1 = - coef[j][1]
            A = np.exp(coef[j][2])

            for i in range(len(yline[j])):
                w = cov.calculate_weigth(y[j], weigth_type='gussian', mu=yline[j][i], sig=GaussianWidth)
                norm[i], slope[i], sig[i] = cov.linear_regression(y[j]*np.log(10), (x[j]-xp)*np.log(10), weight=w)

            xs = 1.0 / beta_2
            Ap = A * np.exp(beta_1**2 * xs / 2.0)
            Ap *= np.sqrt(2.0 * np.pi) * (sig * np.sqrt(xs)) / np.sqrt(sig**2 + xs)
            n_pred += [Ap * np.exp(-(beta_1*xs + slope*yline[j]*np.log(10) + norm)**2 / (2.0 * (sig**2 + xs)))]
            print n_pred

        return n_pred

    def __predict_expected_number_count_obs_method2(self, x, xline, y, yline, labels, xp=14, yp=13, deg=2):

        cov = covariance_package()
        n_pred = []

        n = self.number_count_vector(x, xline, labels)
        coef = self.fit_polynomial(n, xline, xp=xp, deg=deg)

        for j in range(4):

            p = np.poly1d(coef[j])

            slope = np.zeros(len(yline[j]))
            norm = np.zeros(len(yline[j]))
            sig = np.zeros(len(yline[j]))

            A = np.exp(p(xline[j] - xp))

            for i in range(len(yline[j])):
                w = cov.calculate_weigth(y[j], weigth_type='gussian', mu=yline[j][i], sig=GaussianWidth)
                norm[i], slope[i], sig[i] = cov.linear_regression(y[j], (x[j]-xp), weight=w)

            mu = np.log(10.0)*(slope*yline[j] + norm)
            n_pred += [np.exp(p(mu))]

            print n_pred

        return n_pred
    """

    def __predict_expected_number_count_obs_method1(self, x, xline, y, yline, labels, xp=14, yp=13, deg=2):


        cov = covariance_package()
        n_pred = []

        n = self.number_count_vector(x, xline, labels)
        coef = self.fit_polynomial(n, xline, xp=xp, deg=deg)

        for j in range(4):

            slope = np.zeros(len(yline[j]))
            norm = np.zeros(len(yline[j]))
            sig = np.zeros(len(yline[j]))

            for i in range(len(yline[j])):
                w = cov.calculate_weigth(y[j], weigth_type='gussian', mu=yline[j][i], sig=GaussianWidth)
                norm[i], slope[i], sig[i] = cov.linear_regression(y[j], (x[j]-xp), weight=w)

            mu = slope*yline[j] + norm
            A, beta = self.__mass_function_local(mu, coef[j], xp=0)
            """
            for i in range(len(mu)):
                w = self.cov.calculate_weigth(x[j], weigth_type='gussian', mu=mu[i]+xp, sig=self.GaussianWidth)
                norm[i], slope[i], sig[i] = self.cov.linear_regression(x[j], y[j], weight=w)
            n_pred += [A / slope * np.exp(- sig**2 * beta / slope**2)]
            """
            n_pred += [A * slope * np.exp(- sig**2 * beta)]

        return n_pred

    def __predict_expected_number_count_obs_method2(self, x, xline, y, yline, labels, xp=14, yp=13, deg=2):

        n_obs = self.number_count_vector(y, yline, labels)
        coef = self.fit_polynomial(n_obs, yline, xp=yp, deg=deg)
        n_pred = self.expected_number_count(yline, coef, yp)

        return n_pred

    def __predict_expected_number_count_obs_method3(self, x, y, n, xline, xexp, xp, deg):

        coef = self._fit_polynomial(n, xline, xp=xp, deg=deg)

        A, beta = self.__mass_function_local(xexp, coef, xp=14.0)

        slope = []
        sigma = []

        for i in range(len(xexp)):
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xexp[i], sig=self.GaussianWidth)
            intercept, slopel, sigl = self.cov.linear_regression(x, y, weight=w)
            slope += [slopel]
            sigma += [sigl]

        sigma = np.array(sigma)
        slope = np.array(slope)

        return A / slope * np.exp(- sigma**2 * beta / slope**2)

    def predict_number_count_obs(self, x, xline, y, yline, labels, colors, xp=14, yp=13, deg=2):
        '''
        WARNING: The code is wnot working for deg /= 2.
        :param x:
        :param xline:
        :param y:
        :param yline:
        :param labels:
        :param colors:
        :param xp:
        :param yp:
        :param deg:
        :return:
        '''

        n = self.number_count_vector(y, yline, labels)

        n_pred = self.__predict_expected_number_count_obs_method1(x, xline, y, yline, labels, xp=xp, yp=yp, deg=deg)
        n_fit = self.__predict_expected_number_count_obs_method2(x, xline, y, yline, labels, xp=xp, yp=yp, deg=deg)


        sns.set_style("white")
        plt.grid(True)

        # Mass Function Plot
        for i in range(4):
            if i == 1: continue
            plt.plot(yline[i], n_pred[i], ':', c=colors[i])
            plt.plot(yline[i], n_fit[i], '-', c=colors[i], label=labels[i])

        for i in range(len(n)):
            if i == 1: continue
            yplot = (yline[i][1:] + yline[i][:-1]) / 2.0
            plt.semilogy(yplot, n[i], 'o', c=colors[i])


        plt.xlabel(r'$\log(M_{\rm gas, 500c}  [{\rm M}_{\odot}])$', size=20)
        plt.ylabel(r'Number Count [ $1 / d \log (M)$  ]', size=18)

        self._set_legend()
        self._set_ticksize(size=16)

        plt.xlim([12, 13.5])
        plt.ylim([10, 20000])

        plt.savefig(fdir_plot + 'Nobs.png', bbox_inches='tight', dpi=400)


class number_count_class_old():

    def __init__(self):

        self.nBootstrap = 100
        self.GaussianWidth = 0.2
        self.cov = covariance_package()

    def number_count_vector(self, x, xline, labels):

        number_count = []

        for i in range(len(labels)):

            xlinep = xline[i]
            dx = xlinep[1] - xlinep[0]
            xp = x[i]

            ncp = []

            for xmin, xmax in zip(xlinep[:-1], xlinep[1:]):
                ncp += [np.sum((xp>xmin) * (xp <= xmax))]

            number_count += [np.array(ncp) / dx]

        return number_count

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

    def __predict_expected_number_count_obs_method1(self, x, y, yline):

        nobs = []
        xexp = []
        dy = yline[1] - yline[0]
        for ymin, ymax in zip(yline[:-1], yline[1:]):
            mask = (y > ymin) * (y <= ymax)
            nobs += [np.sum(mask)]
            xexp += [np.mean(x[mask])]

        return np.array(nobs)/dy, np.array(xexp)

    def __predict_expected_number_count_obs_method2(self, x, y, n, xline, xexp, xp, deg):

        coef = self._fit_polynomial(n, xline, xp=xp, deg=deg)

        A, beta = self.__mass_function_local(xexp, coef, xp=14.0)

        slope = []
        sigma = []

        for i in range(len(xexp)):
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xexp[i], sig=self.GaussianWidth)
            intercept, slopel, sigl = self.cov.linear_regression(x, y, weight=w)
            slope += [slopel]
            sigma += [sigl]

        sigma = np.array(sigma)
        slope = np.array(slope)

        return A / slope * np.exp(- sigma**2 * beta / slope**2)

    def __predict_expected_number_count_obs_method3(self, x, y, n, xline, xexp, xp, deg):

        coef = self._fit_polynomial(n, xline, xp=xp, deg=deg)

        A, beta = self.__mass_function_local(xexp, coef, xp=14.0)

        slope = []
        sigma = []

        for i in range(len(xexp)):
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xexp[i], sig=self.GaussianWidth)
            intercept, slopel, sigl = self.cov.linear_regression(x, y, weight=w)
            slope += [slopel]
            sigma += [sigl]

        sigma = np.array(sigma)
        slope = np.array(slope)

        return A / slope * np.exp(- sigma**2 * beta / slope**2)

    def predict_expected_number_count_obs(self, x, y, number, xline, ylim, xcut=14.0, nybins=10,
                                          labels=None, colors=None, xp=14.0, deg=3):

        for i in range(len(number)):

            yline = np.linspace(ylim[0], ylim[1], nybins+1)

            nobs, xexp = self.__predict_expected_number_count_obs_method1(x[i], y[i], yline)
            nglob = self.__predict_expected_number_count_obs_method2(x[i], y[i], number[i], xline[i], xexp, xp, deg)

            yline = (yline[1:] + yline[:-1]) / 2.0
            plt.semilogy(yline, nobs, 'o', c=colors[i])
            plt.semilogy(yline, nglob, '-', c=colors[i], label=labels[i])

        plt.xlabel(r'$\log_{10}(M_{\rm gas}  [M_{\odot}])$', size=18)
        plt.ylabel(r'Number Count', size=15)

        # self._set_legend()
        self._set_ticksize(size=14)
        #plt.savefig(fdir_plot + 'number_count.pdf', bbox_inches='tight', dpi=400)
        plt.show()

    def __predict_expected_mass_method1(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            w = self.cov.calculate_weigth(y, weigth_type='gussian', mu=yline[j], sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(y, x, weight=w)
            xline_exp[j] = slope * yline[j] + intercept

        return xline_exp

    def __predict_expected_mass_method2(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            intercept, slope, sig = self.cov.linear_regression(x, y)
            xline_exp[j] = (yline[j] - intercept) / slope

        return xline_exp

    def __predict_expected_mass_method3(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            xp = np.median(x[(y<yline[j]+0.1)*(y>yline[j]-0.1)])
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xp, sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(x, y, weight=w)
            xline_exp[j] = (yline[j] - intercept) / slope

        return xline_exp

    def __predict_expected_mass_method2(self, x, y, yline):

        xline_exp = np.zeros(len(yline))

        for j in range(len(xline_exp)):
            intercept, slope, sig = self.cov.linear_regression(x, y)
            xline_exp[j] = (yline[j] - intercept) / slope

        return xline_exp

    def __predict_expected_mass_method4(self, x, y, yline, coef, xp):

        xline_exp = np.zeros(len(yline))

        deg = len(coef)-1
        c = coef
        c = [c[j] * (deg - j) for j in range(deg)]
        p = np.poly1d(c)

        for j in range(len(xline_exp)):
            xl = np.median(x[(y<yline[j]+0.1)*(y>yline[j]-0.1)])
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xl, sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(x, y, weight=w)
            xline_exp[j] = (yline[j] - intercept) / slope

            beta_1 = p(xl - xp)
            xline_exp[j] += beta_1 * sig**2 / slope**2

        return xline_exp

    def __predict_expected_mass_method5(self, x, y, yline, coef, xp):

        xline_exp = np.zeros(len(yline))

        deg = len(coef)-1
        c = coef
        c = [c[j] * (deg - j) for j in range(deg)]
        p1 = np.poly1d(c)

        deg = len(c)-1
        c = [c[j] * (deg - j) for j in range(deg)]
        p2 = np.poly1d(c)

        for j in range(len(xline_exp)):
            xl = np.median(x[(y<yline[j]+0.1)*(y>yline[j]-0.1)])
            w = self.cov.calculate_weigth(x, weigth_type='gussian', mu=xl, sig=GaussianWidth)
            intercept, slope, sig = self.cov.linear_regression(x, y, weight=w)
            xline_exp[j] = (yline[j] - intercept) / slope

            beta_1 = p1(xl - xp)
            beta_2 = p2(xl - xp)
            xline_exp[j] += beta_1 * sig**2 / slope**2 / (1.0 - beta_2 * sig**2 / slope**2)

        return xline_exp

    def predict_expected_mass(self, x, y, coef, xline, ylim, nybins=10, loc=2,
                              labels=None, colors=None, xp=14.0, xlabel=None, ylabel=None):

        #c = ["#E56124", "#E53E30", "#7F2353", "#F911FF"]
        c = ["#33cc33", "#ff3300", "#0000ff", "#a55927"]

        i = 0

        yline = np.linspace(ylim[0], ylim[1], nybins+1)

        xline_exp1 = self.__predict_expected_mass_method1(x[i], y[i], yline)
        xline_exp2 = self.__predict_expected_mass_method2(x[i], y[i], yline)
        xline_exp3 = self.__predict_expected_mass_method3(x[i], y[i], yline)
        xline_exp4 = self.__predict_expected_mass_method4(x[i], y[i], yline, coef[i], xp)
        xline_exp5 = self.__predict_expected_mass_method5(x[i], y[i], yline, coef[i], xp)

        sns.set_style("white")
        plt.grid(True)

        fig1 = plt.figure(1)

        plt.clf()

        # Plot Data-model
        frame1 = fig1.add_axes((.1, .5, .8, .6))

        # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        mask = x[i] < 13.6
        plt.scatter(y[i][mask], x[i][mask], s=0.5, color=colors[i], alpha=0.08)
        plt.scatter(y[i][~mask], x[i][~mask], s=0.5, color=colors[i], alpha=0.18)
        plt.plot(yline, xline_exp1, color=colors[i], linestyle='-', linewidth=1.3, label='Fit')
        plt.plot(yline, xline_exp2, color=c[0], linestyle='-', linewidth=1.3, label='Power-law prediction')
        plt.plot(yline, xline_exp3, color=c[1], linestyle='-', linewidth=1.3, label='Local power-law prediction')
        plt.plot(yline, xline_exp4, color=c[2], linestyle='-', linewidth=1.3, label='Local power-law prediction with MF correction')
        # plt.plot(yline, xline_exp5, color=c[2], linestyle='-', linewidth=1.3, label='Local power-law prediction with MF correction')

        legend = plt.legend(loc=loc, fontsize=12, fancybox=True, shadow=True)

        frame = legend.get_frame()
        legend.set_frame_on(True)
        frame.set_facecolor('white')

        frame1.set_ylim(13, 15)
        frame1.set_xlim(11.5, 13.7)

        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        frame1.set_yticklabels([' ', 13.5, 14, 14.5, 15])  # Remove x-tic labels for the first frame
        frame1.set_ylabel(ylabel, size=18)
        plt.grid()

        # Residual plot
        frame2 = fig1.add_axes((.1, .1, .8, .4))
        plt.plot([11,15], [0, 0], color='k', linestyle='-', linewidth=1.6)
        plt.plot([11,15], [0.01, 0.01], color='k', linestyle='--', linewidth=0.8)
        plt.plot([11,15], [-0.01, -0.01], color='k', linestyle='--', linewidth=0.8)
        plt.plot(yline, np.log(10.0)*(xline_exp2 - xline_exp1), color=c[0], linestyle='-', linewidth=2.2)
        plt.plot(yline, np.log(10.0)*(xline_exp3 - xline_exp1), color=c[1], linestyle='-', linewidth=2.2)
        plt.plot(yline, np.log(10.0)*(xline_exp4 - xline_exp1), color=c[2], linestyle='-', linewidth=2.2)
        # plt.plot(yline, (xline_exp5 - xline_exp1), color=c[3], linestyle='-', linewidth=2.2)

        #frame2.set_yticklabels([-0.1, -0.05, ' ', 0.0, ' ', 0.05])  # Remove x-tic labels for the first frame
        frame2.set_xlabel(xlabel, size=18)
        # frame2.set_ylabel(r'$\delta (\log M)$')
        frame2.set_ylabel(r'$\%$ Bias', size=18)

        frame2.set_ylim(-0.10, 0.10)
        frame2.set_xlim(11.5, 13.7)

        plt.grid()
        # plt.show()
        plt.savefig('')

    def _set_legend(self):

        legend = plt.legend(loc=1, fontsize=12, fancybox=True, shadow=True)
        frame = legend.get_frame()
        legend.set_frame_on(True)
        frame.set_facecolor('white')
        plt.setp(legend.get_title(), fontsize=12)

    def _set_ticksize(self, size=16, minor_size=8):

        plt.tick_params(axis='both', which='major', labelsize=size)
        plt.tick_params(axis='both', which='minor', labelsize=minor_size)

    def plot_actual_vs_fit(self, xline, n, labels=None, colors=None, xp=14.0, deg=3):

        sns.set_style("white")
        plt.grid(True)

        coef = self.fit_polynomial(n, xline, xp=xp, deg=deg)
        nexp = self.expected_number_count(xline, coef, xp=xp)

        for i in range(len(n)):
            if i == 1: continue
            x = (xline[i][1:] + xline[i][:-1]) / 2.0
            plt.semilogy(x, n[i], 'o', c=colors[i])
            plt.semilogy(xline[i], nexp[i], '-', c=colors[i], label=labels[i])

        plt.xlabel(r'$\log(M_{500c}  [{\rm M}_{\odot}])$', size=20)
        plt.ylabel(r'Number Count', size=18)

        self._set_legend()
        self._set_ticksize(size=16)
        # plt.savefig(fdir_plot + 'number_count.pdf', bbox_inches='tight', dpi=400)

