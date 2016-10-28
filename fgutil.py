import functools

import numpy as np

from scipy.misc import comb

from sklearn.decomposition import PCA


def inv_pca(X, Y, pca, n_mode):
    return pca.inverse_transform(Y[:, :])


def inv_pca_sub(X, Y, pca, n_submode):
    return X - inv_pca(X, Y, pca, n_submode)


def inv_pca_complex(X_real, X_imag, Y_real, Y_imag, pca_real, pca_imag, n_mode):
    Xrec_real = inv_pca(X_real, Y_real, pca_real, n_mode)
    Xrec_imag = inv_pca(X_imag, Y_imag, pca_imag, n_mode)

    return Xrec_real + 1j * Xrec_imag


def inv_pca_sub_complex(X_real, X_imag, Y_real, Y_imag, pca_real, pca_imag, n_submode):
    Xsub_real = inv_pca_sub(X_real, Y_real, pca_real, n_submode)
    Xsub_imag = inv_pca_sub(X_imag, Y_imag, pca_imag, n_submode)

    return Xsub_real + 1j * Xsub_imag


def alm_pca_fit(alm, n_cmpt):
    X_real = alm.T.real
    X_imag = alm.T.imag
    pca_real = PCA(n_components=n_cmpt, whiten=True)
    pca_imag = PCA(n_components=n_cmpt, whiten=True)

    Y_real = pca_real.fit_transform(X_real)
    Y_imag = pca_imag.fit_transform(X_imag)

    print 'PCA: Percentage of variance explained by each of the selected components:'
    print pca_real.explained_variance_ratio_

    X_inv = functools.partial(inv_pca_complex, X_real, X_imag, Y_real, Y_imag, pca_real, pca_imag)

    alm_pca_fitted = X_inv(n_cmpt).T

    return alm_pca_fitted


def bernstein_poly(i, n, x):
    return comb(n, i) * (x ** (n - i)) * (1 - x)**i


def poly_fit(x, y, noiserms, deg):
    C_Dinv = np.diag(1 / noiserms ** 2)
    A = np.vstack([x ** k for k in range(deg + 1)]).T

    lhs = np.dot(np.dot(A.T, C_Dinv), A)
    rhs = np.dot(np.dot(A.T, C_Dinv), y)

    s = np.linalg.solve(lhs, rhs)
    y_s = np.dot(A, s)

    cov_sigma = np.sqrt(np.diag(np.linalg.inv(lhs)))

    return s, y_s, cov_sigma


def bernstein_fit(x, y, noiserms, deg):
    ber_basis = []
    for j in range(deg):
        for i in range(j + 1):
            ber_basis.append(bernstein_poly(i, j, x))

    C_Dinv = np.diag(1 / noiserms ** 2)
    A = np.vstack(ber_basis).T

    lhs = np.dot(np.dot(A.T, C_Dinv), A)
    rhs = np.dot(np.dot(A.T, C_Dinv), y)

    s = np.linalg.solve(lhs, rhs)
    cov_sigma = np.sqrt(np.diag(np.linalg.inv(lhs)))
    y_s = np.dot(A, s)

    return s, y_s, cov_sigma


def powerlaw_fit(x, y, noiserms, deg, bernstein=False):
    if x[0] > 0:
        x = x / x[0]
    sgn = np.sign(np.mean(y))
    y_m = y * sgn
    offset = - 10 * min(np.min(y), -0.05)
    y_m += offset

    if bernstein:
        s, y_s, cov_sigma = bernstein_fit(np.log(x), np.log(y_m), noiserms, deg)
    else:
        s, y_s, cov_sigma = poly_fit(np.log(x), np.log(y_m), noiserms, deg)
    y_s = sgn * (np.exp(y_s) - offset)

    return y_s


def powerlaw_fit_bernstein(x, y, noiserms, deg):
    return powerlaw_fit(x, y, noiserms, deg, bernstein=True)


def alm_poly_fit(freqs, alm, noiserms, deg, fit_fct=powerlaw_fit):
    res = fit_fct(freqs, alm, noiserms, deg)

    if len(res) == 3:
        res = res[1]

    return res
