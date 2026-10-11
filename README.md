# ExoRM

A Python package for estimating exoplanet mass from radius using probabilistic models built with PyMC.

**Status:** Experimental; no longer actively developed.

## Installation

```bash
python -m pip install ExoRM
```

Requires Python 3.9 or later.

## Usage

To fetch the current data from the NASA Exoplanet Archive and fit a model:

```python
from ExoRM import get_data, initialize_model, ExoRM

get_data()
initialize_model(kwargs={"cores": 4})

erm = ExoRM()
erm.load_trace()

# Input: radius in Earth radii
# Output: estimated mass and lower/upper bounds in Earth masses
mass, lower, upper = erm.predict_full_linear([1.0])
```

**Note:** Fitting a model runs PyMC sampling and may take some time. The fitted model and its inputs are saved in ExoRM's user-data directory.

See [`example.ipynb`](example.ipynb) for an example using the model and [`comparison.ipynb`](comparison.ipynb) for comparisons with Forecaster.

## Models

The current implementation includes three model types:

- `nn`: A probabilistic neural network.
- `gp`: A Gaussian process.
- `pwlf`: A piecewise-linear model.

The default model is `nn`. The package also includes `ForecasterRM`, an implementation of the Forecaster mass-radius relationship for comparison.

Predictions are available in logarithmic space through `predict()` and `predict_full()`, or in linear Earth units through `predict_linear()` and `predict_full_linear()`. The latter methods also support uncertainty intervals.

## Repository Contents

| File | Description |
|---|---|
| `ExoRM/` | Package implementation |
| `example.ipynb` | Example usage and predictions |
| `comparison.ipynb` | Comparison with Forecaster |
| `probabilistic.ipynb` | Probabilistic model experiments |
| `exoplanet_data.csv` | Exoplanet data used in the analysis |
| `best_inputs.pkl` | Saved model inputs |
| `best_trace.nc` | Saved posterior trace |
| `best_linear.nc` | Saved trace for the linear model |

Earlier development, including spline experiments, can be found in the Git history.

## Limitations

ExoRM was developed to explore mass-radius modeling and to make the resulting implementation available as a reusable Python package. The comparisons with Forecaster were mixed: ExoRM performed better under some comparisons but had larger errors in others.

The results do not establish that ExoRM is generally more accurate than Forecaster. Use it for exploration and testing rather than as a validated replacement for established models.

## References

- Chen & Kipping, [Forecaster](https://github.com/chenjj2/forecaster)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## License

MIT License. See [`LICENSE`](LICENSE).
