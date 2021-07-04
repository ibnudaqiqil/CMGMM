'''
Created on Sep 13, 2016

@author: Juan Manuel Acevedo Valle
'''
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib._color_data as mcd

from sklearn import mixture
from models.CMGMM import CMGMM
from models.IGMM import IGMM

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold','darkorange','aqua','crimson'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# Fit a Gaussian mixture with EM using five components
gmm = IGMM(min_components=2, max_components=6)
gmm.train(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Incremental Gaussian Mixture')


    # Model computed with three Gaussians
n_samples2 = 150
C2 = np.array([[0., -0.1], [0.2, 0.4]])
X2 = np.r_[np.dot(np.random.randn(n_samples2, 2), 0.5 * C),
           .7 * np.random.randn(n_samples, 2) + np.array([-5, 4]),
           .2 * np.random.randn(n_samples2, 2) + np.array([-2, 1]),
           .5 * np.random.randn(n_samples2, 2) + np.array([1, 3]),
           .5 * np.random.randn(n_samples2, 2) + np.array([1, 3]),
           .4 * np.random.randn(n_samples, 2) + np.array([-2, -0.5])]

gmm.train(X2)

final = np.concatenate((X, X2), axis=0)
plot_results(final, gmm.predict(final), gmm.means_, gmm.covariances_, 1,
             'Model After Data Drift')

plt.show()
