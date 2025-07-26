from fastapi.testclient import TestClient
import pytest
from main import app

client = TestClient(app)


def test_linear_regression():
    data: dict[str, list[int] | int] = {
        "x": [0, 1, 2, 3],
        "y": [1, 3, 5, 7],
        "degree": 1,
    }
    response = client.post("/regression", json=data)
    assert response.status_code == 200
    result = response.json()

    assert "coefficients" in result
    assert "r_squared" in result
    assert round(result["r_squared"], 2) == 1.0


def test_linear_correctness():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    data: dict[str, list[int] | int] = {
        "x": x,
        "y": y,
        "degree": 1,
    }
    response = client.post("/regression", json=data)
    assert response.status_code == 200
    result = response.json()

    assert "coefficients" in result
    assert len(result["coefficients"]) == 2  # slope and intercept
    print(result)
    assert result["coefficients"][1] == pytest.approx(2.0)  # type: ignore
    assert result["coefficients"][0] == pytest.approx(0.0)  # type: ignore
    assert result["r_squared"] == pytest.approx(1.0)  # type: ignore
    for wald in result["wald_test"]:
        assert "coefficient" in wald
        assert "std_error" in wald
        assert "t_stat" in wald
        assert "p_value" in wald
        assert wald["p_value"] == pytest.approx(0.0)  # type: ignore


def test_polynomial_regression():
    data: dict[str, list[int] | int] = {
        "x": [0, 1, 2, 3, 4],
        "y": [4, 1, 0, 1, 4],
        "degree": 2,
    }
    response = client.post("/regression", json=data)
    assert response.status_code == 200
    result = response.json()

    assert "coefficients" in result
    assert len(result["coefficients"]) == 3  # quadratic, linear, and constant terms
    assert result["r_squared"] == pytest.approx(1.0)  # type: ignore

    for wald in result["wald_test"]:
        assert "coefficient" in wald
        assert "std_error" in wald
        assert "t_stat" in wald
        assert "p_value" in wald
        assert wald["p_value"] == pytest.approx(0.0)  # type: ignore
        assert wald["std_error"] > 0.0  # Standard error should be positive


def test_invalid_length():
    data: dict[str, list[int] | int] = {
        "x": [1, 2, 3],
        "y": [1, 2],
        "degree": 1,
    }
    response = client.post("/regression", json=data)
    assert response.status_code == 400
    assert response.json() == {"detail": "Length of x and y must match."}


def test_singular_matrix():
    data: dict[str, list[int] | int] = {
        "x": [1, 1, 1],
        "y": [1, 2, 3],
        "degree": 1,
    }
    response = client.post("/regression", json=data)
    assert response.status_code == 400
    assert response.json() == {"detail": "Design matrix XᵀX is singular."}
