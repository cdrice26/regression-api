from fastapi.testclient import TestClient
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
