
import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt
from scipy import stats
from scr.covariance_package import covariance_package

fdir_plot = './plots/'
nBootstrap = 100
GaussianWidth = 0.2


def plot_regression(ax, x, y, xline, labels, number, **kwargs):

    cov = covariance_package()

    yline = np.zeros(len(xline))

    ax.scatter(x, y, s=0.5, color=kwargs['color'], alpha=kwargs['alpha'][number])

    if kwargs['fit_model'][number] == 'Local':

        for i in range(len(xline)):
            w = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig = GaussianWidth)
            intercept, slope, sig = cov.linear_regression(x, y, weight=w)
            yline[i] = slope * xline[i] + intercept

    elif kwargs['fit_model'][number] == 'Global':

        intercept, slope, sig = cov.linear_regression(x, y)
        yline = slope * xline + intercept

    ax.plot(xline, yline, color=kwargs['color'],  linestyle='-', linewidth=1.3, label=labels)


def plot_mass_evolution(ax, x, y, xline, labels, number, **kwargs):

    cov = covariance_package()

    if kwargs['fit_model'][number] == 'Local':

        lenX = len(x)
        slope = np.zeros([nBootstrap, len(xline)])
        norm = np.zeros([nBootstrap, len(xline)])
        sig = np.zeros([nBootstrap, len(xline)])

        for iBoot in range(nBootstrap):

            resample_i = np.floor(np.random.rand(lenX) * lenX).astype(int)

            for i in range(len(xline)):

                if kwargs['filter_type'] == 'gaussian':
                    w = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=GaussianWidth)
                elif kwargs['filter_type'] == 'uniform':
                    w = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i]-.1, xmax=xline[i]+0.1)

                norm[iBoot, i], slope[iBoot, i], sig[iBoot, i] =\
                    cov.linear_regression(x[resample_i], y[resample_i], weight=w[resample_i])

        ax[0].plot(xline, np.mean(slope, axis=0), color=kwargs['color'], linestyle='-', linewidth=1.3, label=labels)
        ax[0].fill_between(xline, np.percentile(slope, 16, axis=0), np.percentile(slope, 84, axis=0),
                           facecolor=kwargs['color'], alpha=0.4, label=None)
        ax[1].plot(xline, np.log(10.0)*np.mean(sig, axis=0), color=kwargs['color'], linestyle='-', linewidth=1.3, label=labels)
        ax[1].fill_between(xline, np.log(10.0)*np.percentile(sig, 16, axis=0), np.log(10.0)*np.percentile(sig, 84, axis=0),
                           facecolor=kwargs['color'], alpha=0.4, label=None)
        ax[0].grid(True)
        ax[1].grid(True)

    elif kwargs['fit_model'][number] == 'Global':
        lenX = len(x)
        intercept = np.zeros(nBootstrap)
        slope = np.zeros(nBootstrap)
        sig = np.zeros(nBootstrap)
        for iBoot in range(nBootstrap):
            resample_i = np.floor(np.random.rand(lenX) * lenX).astype(int)
            intercept[iBoot], slope[iBoot], sig[iBoot] = cov.linear_regression(x[resample_i], y[resample_i])

        ax[0].errorbar(np.mean(x), np.mean(slope), yerr=np.std(slope),
                       color=kwargs['color'], marker=kwargs['marker'], label=labels)
        ax[1].errorbar(np.mean(x), np.log(10.0)*np.mean(sig), yerr=np.log(10.0)*np.std(sig),
                       color=kwargs['color'], marker=kwargs['marker'], label=labels)

        # ax[0].plot(np.mean(x)-0.3, slope, 'o', color=kwargs['color'], label=labels)
        # ax[1].plot(np.mean(x)-0.3, np.log(10.0)*sig, 'o', color=kwargs['color'], label=labels)


def plot_residual(ax, x, y, xline, labels, number, **kwargs):

    cov = covariance_package()
            
    dy = []

    if kwargs['fit_model'][number] == 'Local':

        for i in range(len(xline) - 1):

            if kwargs['filter_type'] == 'gaussian':
                w1 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=GaussianWidth)
                w2 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i + 1], sig=GaussianWidth)

            elif kwargs['filter_type'] == 'uniform':
                w1 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i] - .1, xmax=xline[i] + 0.1)
                w2 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i + 1] - .1, xmax=xline[i + 1] + 0.1)

            intercept, slope, _ = cov.linear_regression(x, y, weight=w1)
            yline1 = slope * xline[i] + intercept
            intercept, slope, _ = cov.linear_regression(x, y, weight=w2)
            yline2 = slope * xline[i + 1] + intercept

            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])
            intercept = - slope * xline[i] + yline1

            mask = (x > xline[i]) * (x < xline[i + 1])

            std = np.std(y[mask] - slope * x[mask] - intercept)
            dy += list((y[mask] - slope * x[mask] - intercept)/std)

    elif kwargs['fit_model'][number] == 'Global':

        intercept, slope, sig = cov.linear_regression(x, y)

        std = np.std(y - slope * x - intercept)
        dy = list((y - slope * x - intercept)/std)

    res, bin_edges = np.histogram(dy, bins=kwargs['bins']/4, range=kwargs['xrange'], normed=True)
    Mean, Variance, Skewness, Kurtosis = stats.describe(dy)[2:]

    bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2.
    ax.plot(bin_edges, res, color=kwargs['color'],  linestyle='-', linewidth=1.8,
            label=r'%6.3f  %5.2f  %5.2f'%(Mean, Skewness, Kurtosis))
    # return dy


def plot_qq(ax, x, y, xline, labels, number, **kwargs):
    cov = covariance_package()

    dy = []

    if kwargs['fit_model'][number] == 'Local':

        for i in range(len(xline) - 1):

            if kwargs['filter_type'] == 'gaussian':
                w1 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=GaussianWidth)
                w2 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i + 1], sig=GaussianWidth)

            elif kwargs['filter_type'] == 'uniform':
                w1 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i] - .1, xmax=xline[i] + 0.1)
                w2 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i + 1] - .1, xmax=xline[i + 1] + 0.1)

            intercept, slope, _ = cov.linear_regression(x, y, weight=w1)
            yline1 = slope * xline[i] + intercept
            intercept, slope, _ = cov.linear_regression(x, y, weight=w2)
            yline2 = slope * xline[i + 1] + intercept

            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])
            intercept = - slope * xline[i] + yline1

            mask = (x > xline[i]) * (x < xline[i + 1])

            std = np.std(y[mask] - slope * x[mask] - intercept)
            dy += list((y[mask] - slope * x[mask] - intercept) / std)

    elif kwargs['fit_model'][number] == 'Global':

        intercept, slope, sig = cov.linear_regression(x, y)

        std = np.std(y - slope * x - intercept)
        dy = list((y - slope * x - intercept) / std)

    res, bin_edges = np.histogram(dy, bins=kwargs['bins'] / 4, range=kwargs['xrange'], normed=True)
    Mean, Variance, Skewness, Kurtosis = stats.describe(dy)[2:]

    dy.sort()
    norm = npr.normal(Mean, 1, len(dy))
    norm.sort()

    ax.plot(norm, dy, ".", color=kwargs['color'])

    ax.plot((-5, 5), (-5, 5), 'k--', linewidth=2)


def plot_correlation(ax, x, y, z, xline, labels,  number, **kwargs):

    cov = covariance_package()

    cline = np.zeros(len(xline)-1)

    if kwargs['fit_model'][number] == 'Local':

        for i in range(len(xline)-1):

            mask = (x > xline[i]) * (x < xline[i+1])

            if kwargs['filter_type'] == 'gaussian':
                w1 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=GaussianWidth)
                w2 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i + 1], sig=GaussianWidth)

            elif kwargs['filter_type'] == 'uniform':
                w1 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i] - .1, xmax=xline[i] + 0.1)
                w2 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i + 1] - .1, xmax=xline[i + 1] + 0.1)

            intercept, slope, _ = cov.linear_regression(x, y, weight=w1)
            yline1 = slope * xline[i] + intercept
            intercept, slope, _ = cov.linear_regression(x, y, weight=w2)
            yline2 = slope * xline[i+1] + intercept
            
            slope = (yline2-yline1)/(xline[i+1]-xline[i])
            intercept = - slope * xline[i] + yline1
            
            dy = list(y[mask] - slope * x[mask] - intercept)

            intercept, slope, _ = cov.linear_regression(x, z, weight=w1)
            yline1 = slope * xline[i] + intercept
            intercept, slope, _ = cov.linear_regression(x, z, weight=w2)
            yline2 = slope * xline[i+1] + intercept
            
            slope = (yline2-yline1)/(xline[i+1]-xline[i])
            intercept = - slope * xline[i] + yline1
            
            dz = list(z[mask] - slope * x[mask] - intercept)

            cline[i] = cov.calc_correlation_fixed_x(x, y, z, weight=(w1+w2)/2.)
            #cline[i] = np.corrcoef(dy, dz)[1, 0]

        ax.plot((xline[1:] + xline[:-1]) / 2., cline, color=kwargs['color'], marker=kwargs['marker'],
                linestyle=None, linewidth=0, label=labels)

    if kwargs['fit_model'][number] == 'Global':

        intercept, slope, _ = cov.linear_regression(x, y)
        dy = list(y - slope * x - intercept)

        intercept, slope, _ = cov.linear_regression(x, z)
        dz = list(z - slope * x - intercept)

        cline = np.corrcoef(dy, dz)[1, 0]

        ax.plot(np.mean(x), cline, color=kwargs['color'], marker=kwargs['marker'],
                linestyle=None, linewidth=0, label=labels)

    if kwargs['fit_model'][number] == 'All':

        intercept, slope, _ = cov.linear_regression(x, y)
        dy = list(y - slope * x - intercept)

        intercept, slope, _ = cov.linear_regression(x, z)
        dz = list(z - slope * x - intercept)

        cline = np.corrcoef(dy, dz)[1, 0]

        ax.plot(np.mean(x) - 1.0, cline, color=kwargs['color'], marker=kwargs['marker'],
                linestyle=None, linewidth=0, label=labels)


def plot_correlation_global(ax, x, y, z, xline, labels, number, **kwargs):
    cov = covariance_package()

    if labels == r'Z = 1 (BAHAMAS)':
        return 0

    if kwargs['fit_model'][number] == 'Global':

        nmax = kwargs['nbins']

        for i in range(nmax):
            xmin = min(x) + float(i) * (max(x) - min(x)) / float(nmax)
            xmax = min(x) + float(i+1) * (max(x) - min(x)) / float(nmax)
            mask = (x < xmax) * (x > xmin)
            xp = x[mask]
            yp = y[mask]
            zp = z[mask]
            lenX = len(xp)

            cline = []
            for iBoot in range(1000):
                resample_i = np.floor(np.random.rand(lenX) * lenX).astype(int)
                intercept, slope, _ = cov.linear_regression(xp[resample_i], yp[resample_i])
                dy = list(yp[resample_i] - slope * xp[resample_i] - intercept)

                intercept, slope, _ = cov.linear_regression(xp[resample_i], zp[resample_i])
                dz = list(zp[resample_i] - slope * xp[resample_i] - intercept)

                cline += [np.corrcoef(dy, dz)[1, 0]]

            print labels, np.mean(xp), np.mean(cline), np.mean(xp), np.std(cline)

            # ax.plot(np.mean(xp), np.mean(cline), color=kwargs['color'], marker=kwargs['marker'],
            #         linestyle=None, linewidth=0, label=None)
            ax.errorbar(np.mean(xp), np.mean(cline), yerr=np.std(cline),
                        color=kwargs['color'], marker=kwargs['marker'], label=None)

        ax.plot(np.mean(xp), np.mean(cline), color=kwargs['color'], marker=kwargs['marker'],
                linestyle=None, linewidth=0, label=labels)


def plot_cumulative_correlation(ax, x, y, z, xline, labels,  number, **kwargs):

    cov = covariance_package()

    cline = np.zeros(len(xline)-1)
            
    dy = []
    dz = []

    if kwargs['fit_model'][number] == 'Local':

        for i in range(len(xline)-2, -1, -1):

            mask = (x > xline[i]) * (x < xline[i+1])

            if kwargs['filter_type'] == 'gaussian':
                w1 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=GaussianWidth)
                w2 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i + 1], sig=GaussianWidth)

            elif kwargs['filter_type'] == 'uniform':
                w1 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i] - .1, xmax=xline[i] + 0.1)
                w2 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i + 1] - .1, xmax=xline[i + 1] + 0.1)

            intercept, slope, _ = cov.linear_regression(x, y, weight=w1)
            yline1 = slope * xline[i] + intercept
            intercept, slope, _ = cov.linear_regression(x, y, weight=w2)
            yline2 = slope * xline[i + 1] + intercept

            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])
            intercept = - slope * xline[i] + yline1

            dy += list(y[mask] - slope * x[mask] - intercept)

            intercept, slope, _ = cov.linear_regression(x, z, weight=w1)
            yline1 = slope * xline[i] + intercept
            intercept, slope, _ = cov.linear_regression(x, z, weight=w2)
            yline2 = slope * xline[i + 1] + intercept

            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])
            intercept = - slope * xline[i] + yline1

            dz += list(z[mask] - slope * x[mask] - intercept)

            cline[i] = np.corrcoef(dy, dz)[1, 0]

        ax.plot((xline[1:] + xline[:-1]) / 2., cline, color=kwargs['color'], marker=kwargs['marker'],
                linestyle=None, linewidth=0, label=labels)

    if kwargs['fit_model'][number] == 'Global':

        intercept, slope, _ = cov.linear_regression(x, y)
        dy = list(y - slope * x - intercept)

        intercept, slope, _ = cov.linear_regression(x, z)
        dz = list(z - slope * x - intercept)

        cline = np.corrcoef(dy, dz)[1, 0]

        ax.plot(np.mean(x) - 0.3, cline, color=kwargs['color'], marker=kwargs['marker'],
                linestyle=None, linewidth=0, label=labels)


"""
        if kwargs['filter_type'] == 'gaussian':
            w1 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i], sig=0.1)
            w2 = cov.calculate_weigth(x, weigth_type='gussian', mu=xline[i+1], sig=0.1)
        elif kwargs['filter_type'] == 'uniform':
            w1 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i]-.1, xmax=xline[i]+0.1)
            w2 = cov.calculate_weigth(x, weigth_type='uniform', xmin=xline[i+1]-.1, xmax=xline[i+1]+0.1)

        intercept, slope, _ = cov.linear_regression(x, y, weight=w1)
        yline1 = slope * xline[i] + intercept
        intercept, slope, _ = cov.linear_regression(x, y, weight=w2)
        yline2 = slope * xline[i+1] + intercept
            
        slope = (yline2-yline1)/(xline[i+1]-xline[i])
        intercept = - slope*xline[i] + yline1
            
        dy += list(y[mask] - slope * x[mask] - intercept)

        intercept, slope, _ = cov.linear_regression(x, z, weight=w1)
        zline1 = slope * xline[i] + intercept
        intercept, slope, _ = cov.linear_regression(x, z, weight=w2)
        zline2 = slope * xline[i+1] + intercept
            
        slope = (zline2-zline1)/(xline[i+1]-xline[i])
        intercept = - slope*xline[i] + zline1
            
        dz += list(z[mask] - slope * x[mask] - intercept)

        cline[i] = np.corrcoef(dy, dz)[0, 1]

    ax.plot((xline[1:]+xline[:-1])/2., cline, color=kwargs['color'], marker=kwargs['marker'],
            linestyle=None, linewidth=0, label=labels)
"""


def wu_calc(x, y, z, labels, number, **kwargs):

    cov = covariance_package()

    sig_y = []
    sig_z = []
    corr  = []
    if kwargs['fit_model'][number] == 'All':
        print "Lable : ", labels

        intercept, slope, _ = cov.linear_regression(x, y)
        dy = list(y - slope * x - intercept)

        print "Slope (Mgas , Mstar) : ", slope,

        intercept, slope, _ = cov.linear_regression(x, z)
        dz = list(z - slope * x - intercept)
        print slope

        corr = np.corrcoef(dy, dz)[1, 0]

        print "Fractional scatter (Mgas , Mstar) : ", np.std(dy)*np.log(10), np.std(dz)*np.log(10)
        print "correlation ", corr