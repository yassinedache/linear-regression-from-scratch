"""linreg: linear regression from scratch.

Least squares by normal equations (Cholesky) and by Householder QR,
and gradient descent with a study of the step size and of convergence.
"""

from . import data, gradient_descent, least_squares, linalg, metrics

__all__ = ["data", "gradient_descent", "least_squares", "linalg", "metrics"]
__version__ = "1.0.0"
