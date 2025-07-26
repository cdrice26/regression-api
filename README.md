# ChefDeck Python API

## Overview

This is an API created with FastAPI that has a single endpoint `/regression` that takes in arrays of numbers `x` and `y`, as well as the `degree` of the polynomial to use for regression. It generates output with the coefficients of the equation, the $R^2$ value, and output of an ANOVA test on the regression.

## Requirements

- Python 3.12+ (older versions may work, but it has not been tested on versions prior to this)

## Installation

1. Clone the repository:

  ```sh
  git clone https://github.com/cdrice26/regression-api.git
  cd regression-api
  ```
2. Create a virtual environment:
  ```sh
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```

3. Install the required packages:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

To run the application, use the following command:

```sh
fastapi dev src/main.py
```

Once the server is running, you can access the API documentation at:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

This will show you the available endpoint and how to use it, but essentially, it is at `/regression` and takes in a JSON payload like so:
```json
{
  "x": float[],
  "y": float[],
  "degree": int
}
```
and returns a JSON payload of the following format:

```json
{
  "coefficients": float[],
  "correlation_coefficient": float,
  "r_squared": float,
  "test_results": {
    "f_stat": float,
    "p_value": float
  }
}
```

## Testing

To run the tests, use:

```sh
pytest
```

Make sure to have `pytest` installed in your environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Note that third party libraries are bundled with the final project, and remain under their respective licenses - see [third_party_licenses](third_party_licenses) for details. 