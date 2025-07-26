from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from numpy.typing import NDArray

from utils import (
    get_correlation_coefficient,
    get_predictions,
    get_r_squared,
    get_residuals,
    get_ss_residuals,
    get_ss_total,
    get_vander_monde_matrix,
    run_f_test,
    solve_normal_equations,
)

app = FastAPI(title="Regression API")


class RegressionRequest(BaseModel):
    x: List[float]
    y: List[float]
    degree: int = 1


class TestResult(BaseModel):
    f_stat: float
    p_value: float


class RegressionResponse(BaseModel):
    coefficients: List[float]
    correlation_coefficient: float
    r_squared: float
    test_results: TestResult


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
    X: NDArray[np.float64] = get_vander_monde_matrix(x, p)

    # Solve normal equations: beta = (Xᵀ X)⁻¹ Xᵀ y
    try:
        beta = solve_normal_equations(X, y)
    except np.linalg.LinAlgError:
        raise HTTPException(status_code=400, detail="Design matrix XᵀX is singular.")

    # Predictions and residuals
    y_pred = get_predictions(X, beta)

    # Pearson correlation coefficient
    correlation_coefficient = get_correlation_coefficient(y, y_pred)
    residuals = get_residuals(y, y_pred)

    # R-squared
    ss_res = get_ss_residuals(residuals)
    ss_tot = get_ss_total(y)
    r_squared = get_r_squared(ss_res, ss_tot)

    # F-test for overall significance
    test_stat, p_value = run_f_test(n, p, ss_res, ss_tot)

    return RegressionResponse(
        coefficients=beta.tolist(),
        correlation_coefficient=correlation_coefficient,
        r_squared=r_squared,
        test_results=TestResult(f_stat=test_stat, p_value=p_value),
    )


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
