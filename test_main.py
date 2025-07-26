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
    assert result["test_results"]["p_value"] == pytest.approx(0.0)  # type: ignore


def test_another_linear_dataset():
    x = [
        1995,
        1996,
        1997,
        1998,
        1999,
        2000,
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
    ]
    y = [
        25.2,
        28.6,
        30.2,
        29.2,
        28.4,
        30.6,
        26.4,
        27.0,
        29.0,
        27.8,
        25.8,
        29.0,
        29.0,
        27.6,
        29.4,
        30.8,
        31.4,
        29.4,
        30.2,
        29.2,
        30.2,
        27.6,
        28.6,
        31.6,
        32.2,
        31.2,
        32.6,
        33.8,
        31.2,
    ]
    data: dict[str, list[int] | list[float] | int] = {
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
    assert result["coefficients"][0] == pytest.approx(-279.5496552)  # type: ignore # intercept
    assert result["coefficients"][1] == pytest.approx(0.153793103)  # type: ignore # slope
    assert result["r_squared"] == pytest.approx(0.41640111)  # type: ignore
    assert result["test_results"]["p_value"] == pytest.approx(0.000157035, rel=1e-3)  # type: ignore


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

    assert result["test_results"]["p_value"] == pytest.approx(0.0)  # type: ignore


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
