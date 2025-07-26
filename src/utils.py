import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_dist  # type: ignore


def get_vander_monde_matrix(x: NDArray[np.float64], p: int) -> NDArray[np.float64]:
    return np.vander(x, N=p, increasing=True).astype(np.float64)


def solve_normal_equations(
    X: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    XtX: NDArray[np.float64] = X.T @ X
    XtX_inv: NDArray[np.float64] = np.linalg.inv(XtX).astype(np.float64)
    return XtX_inv @ (X.T @ y)


def get_predictions(
    X: NDArray[np.float64], beta: NDArray[np.float64]
) -> NDArray[np.float64]:
    return X @ beta


def get_residuals(
    y: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> NDArray[np.float64]:
    return y - y_pred


def get_correlation_coefficient(
    y: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    correlation_matrix = np.corrcoef(y, y_pred)
    return correlation_matrix[0, 1] if correlation_matrix.size > 1 else 0.0


def get_ss_residuals(residuals: NDArray[np.float64]) -> float:
    return float(np.sum(residuals**2))


def get_ss_total(y: NDArray[np.float64]) -> float:
    return np.sum((y - np.mean(y)) ** 2)


def get_r_squared(ss_res: float, ss_tot: float) -> float:
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def run_f_test(n: int, p: int, ss_res: float, ss_total: float) -> tuple[float, float]:
    df_regression = p - 1  # exclude intercept
    df_residual = n - p

    ss_regression = ss_total - ss_res

    ms_regression = ss_regression / df_regression
    ms_residual = ss_res / df_residual

    f_stat = ms_regression / ms_residual
    p_value: int = 1.0 - f_dist.cdf(f_stat, df_regression, df_residual)  # type: ignore

    # Check for negative infinity
    if f_stat == -float("inf"):
        f_stat = np.finfo(float).min  # Replace negative infinity with min float

    # Replace infinite values with appropriate finite floats
    if f_stat == float("inf"):
        f_stat = np.finfo(float).max  # Replace positive infinity with max float

    return f_stat, p_value
