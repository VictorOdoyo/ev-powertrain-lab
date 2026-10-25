import numpy as np

from .core import integer, stream


def correlated_samples(mean, covariance, count=1000, seed=42):
    from .core import finite

    mu = finite(mean, real=True, ndim=1)
    cov = finite(covariance, real=True, ndim=2)
    if cov.shape != (len(mu), len(mu)) or not np.allclose(cov, cov.T) or np.min(np.linalg.eigvalsh(cov)) < 0:
        raise ValueError("symmetric positive semidefinite covariance required")
    integer(count, 2)
    return stream(seed, "physical-uncertainty").multivariate_normal(mu, cov, count)
