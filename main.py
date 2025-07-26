from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as t_dist  # type: ignore

app = FastAPI(title="Regression API")


class RegressionRequest(BaseModel):
    x: List[float]
    y: List[float]
    degree: int = 1


class CoefficientResult(BaseModel):
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float


class RegressionResponse(BaseModel):
    coefficients: List[float]
    correlation_coefficient: float
    r_squared: float
    wald_test: List[CoefficientResult]


@app.post("/regression", response_model=RegressionResponse)
def fit_regression(req: RegressionRequest) -> RegressionResponse:
    x: NDArray[np.float64] = np.array(req.x, dtype=np.float64)
    y: NDArray[np.float64] = np.array(req.y, dtype=np.float64)
    n: int = x.shape[0]
    degree: int = req.degree
    p: int = degree + 1  # number of regression parameters

    if y.shape[0] != n:
        raise HTTPException(status_code=400, detail="Length of x and y must match.")

    # Build design matrix with polynomial terms up to 'degree'
    X: NDArray[np.float64] = np.vander(x, N=p, increasing=True).astype(np.float64)

    # Solve normal equations: beta = (Xᵀ X)⁻¹ Xᵀ y
    XtX: NDArray[np.float64] = X.T @ X
    try:
        XtX_inv: NDArray[np.float64] = np.linalg.inv(XtX).astype(np.float64)
    except np.linalg.LinAlgError:
        raise HTTPException(status_code=400, detail="Design matrix XᵀX is singular.")

    beta: NDArray[np.float64] = XtX_inv @ (X.T @ y)

    # Predictions and residuals
    y_pred: NDArray[np.float64] = X @ beta
    residuals: NDArray[np.float64] = y - y_pred

    # Pearson correlation coefficient
    correlation_coefficient: float = float(np.corrcoef(y, y_pred)[0, 1])

    # R-squared
    ss_res: float = float(np.sum(residuals**2))
    ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
    r_squared: float = 1.0 - ss_res / ss_tot

    # Wald T-test for each coefficient
    df: int = n - p
    sigma2: float = ss_res / df
    cov_beta: NDArray[np.float64] = sigma2 * XtX_inv
    se_beta: NDArray[np.float64] = np.sqrt(np.diag(cov_beta))

    wald_results: List[CoefficientResult] = []
    for idx in range(p):
        coef: float = float(beta[idx])
        std_err: float = float(se_beta[idx])
        t_stat: float = coef / std_err
        raw_p = t_dist.sf(float(abs(t_stat)), df)  # type: ignore
        p_value: float = float(2.0 * raw_p)
        wald_results.append(
            CoefficientResult(
                coefficient=coef,
                std_error=std_err,
                t_stat=t_stat,
                p_value=p_value,
            )
        )

    return RegressionResponse(
        coefficients=beta.tolist(),
        correlation_coefficient=correlation_coefficient,
        r_squared=r_squared,
        wald_test=wald_results,
    )


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
