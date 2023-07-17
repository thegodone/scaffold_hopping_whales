# ======================================================================================================================
# * Weighted Holistic Atom Localization and Entity Shape (WHALES) descriptors *
#   v. 1, May 2018
# ----------------------------------------------------------------------------------------------------------------------
# This file contains all the necessary files to calculate atom centred mahalanobis descriptors.
# Starting from the 3D coordinates and the partial charges of the molecules, the isolation degree, remoteness
# and their ratio are computed.
# The covariance is centered on each atom and weighted according to the partial charges of the selected surrounding
# atoms.
#
# Francesca Grisoni, May 2018, ETH Zurich & University of Milano-Bicocca, francesca.grisoni@unimib.it
# please cite as: 
#   Francesca Grisoni, Daniel Merk, Viviana Consonni, Jan A. Hiss, Sara Giani Tagliabue, Roberto Todeschini & Gisbert Schneider 
#   "Scaffold hopping from natural products to synthetic mimetics by holistic molecular similarity", 
#   Nature Communications Chemistry 1, 44, 2018.
# ======================================================================================================================
# This version of the code was simplified and vectorized by Guillaume Godin July 2023, dsm-firmenich, Geneva
# it works on numpy 3.9
# ======================================================================================================================
import numpy as np

def docov_vect(x, w):
    n, p = x.shape 
    
    cov = np.zeros((n, p, p))

    type_w = 1 
    
    den = np.sum(np.abs(w)) if type_w == 1 else (n-1)
    
    w_abs = np.abs(w) / den
    x_diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    x_diff_square = x_diff[:, :, np.newaxis, :] * x_diff[:, :, :, np.newaxis]
    
    # caution we need to do it like this!
    w_x_diff_square = w_abs[np.newaxis, :, np.newaxis, np.newaxis] *  x_diff_square
    # we have an extra first dimension do to the w_abs expension just remove it by taking the first element off
    cov = np.sum(w_x_diff_square, axis=1)[0]

    return cov

def domahal_vectorized(x, cov_dict):
    """
    Vectorized function for calculating the atom centred Mahalanobis distance
    between all pairs of atoms when the covariance is given as a dictionary.
    """
    n, p = x.shape  # matrix dimensions
    dist = np.empty((n, n))
    for j in range(n):
        res = x - x[j, :]
        pinv_cov_j = np.linalg.pinv(cov_dict[(j, 1)])
        dist[:, j] = np.einsum('ij,ij->i', np.dot(res, pinv_cov_j), res)
    return dist/p

def domahal_vectorized_vect(x, cov):
    """
    Vectorized function for calculating the atom centred Mahalanobis distance
    between all pairs of atoms when the covariance is given as a dictionary.
    """
    n, p = x.shape  # matrix dimensions
    dist = np.empty((n, n))
    for j in range(n):
        res = x - x[j, :]
        pinv_cov_j = np.linalg.pinv(cov[j])
        dist[:, j] = np.einsum('ij,ij->i', np.dot(res, pinv_cov_j), res)
    return dist/p


def lmahal_vect(x, w):
    """
    main function for calculating the atom-centred mahalanobis distance (ACM), used to compute remoteness and
    isolation degree.

    ====================================================================================================================
    :param
    x(n_at x 3): molecular 3D coordinate matrix
    w(n_at x 1): molecular property to consider
    :return
    res(n_at x 3): atomic descriptors; col 0 = Remoteness, col 1 = Isolation degree, col 2 = Isol/Remoteness

    REF: Todeschini, et al. "Locally centred Mahalanobis distance: a new distance measure with salient features towards
    outlier detection." Analytica chimica acta 787 (2013): 1-9.
    ====================================================================================================================
    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """

    # preliminary
    if len(w) > 0:   # checks whether at least one atom was included
        # do covariance centred on each sample
        cov = docov_vect(x, w)
        dist_vect = domahal_vectorized_vect(x, cov)
        # isolation and remoteness parameters from D
        res = is_rem(dist_vect) 
    else:
        res = np.full((1, 3), -999.0)   # sets missing values

    return res

# ----------------------------------------------------------------------------------------------------------------------

def is_rem(dist): 
    """
    Calculates isolation degree and remoteness from a distance matrix and their ratio.
    ====================================================================================================================
    :param
    dist: atom-centred Mahalanobis
    n: number of compounds
    :returns
    isol: isolation degree (column minimum)
    rem: remoteness (row average)
    ir_ratio: ratio between isolation degree and remoteness
    ====================================================================================================================
    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """
    
    np.fill_diagonal(dist, np.nan)

    isol = np.nanmin(dist, axis=0, keepdims=True).T  # col minimum (transposed for dimensions)
    rem = np.nanmean(dist, axis=1, keepdims=True)  # row average (what if it is zero mean ???)
    ir_ratio = isol / rem  # ratio between isol and rem (transpose for dimensions)

    res = np.hstack((rem, isol, ir_ratio))  # results concatenation

    return res
