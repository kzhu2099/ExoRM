# ExoRM

A Python package for estimating exoplanet mass from radius using probabilistic models built with PyMC.

**Status:** Experimental and no longer actively developed.

I started ExoRM to try different ways of modeling the relationship between exoplanet mass and radius, then compare them with Forecaster.

## Installation

```bash
python -m pip install ExoRM
```

## Usage

To fetch the current data from the NASA Exoplanet Archive and fit a model:

```python
from ExoRM import get_data, initialize_model, ExoRM

get_data()
initialize_model(kwargs={'cores': 4})

erm = ExoRM()
erm.load_trace()

# Input: radius in Earth radii
# Output: estimated mass and lower/upper bounds in Earth masses
mass, lower, upper = erm.predict_full_linear([1.0])
```

Fitting a model runs PyMC sampling and may take some time. The fitted model and its inputs are saved in ExoRM's user-data directory.

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
| `probabilistic.ipynb` | Probabilistic model experiments and other tests |
| `exoplanet_data.csv` | Previous exoplanet data used in the analysis |
| `best_inputs.pkl` | Saved model inputs from best tests |
| `best_trace.nc` | Saved trace for main model |
| `best_linear.nc` | Saved trace for the linear model |

Earlier development, including spline experiments, can be found in earlier PyPI versions.

## Limitations

The comparisons with Forecaster were mixed. ExoRM performed better under some comparisons but had larger errors in others. The results do not establish that ExoRM is generally more accurate than Forecaster.

The data come from the NASA Exoplanet Archive, so rerunning the download may produce a different sample as the catalog changes.

## References

- Chen & Kipping [Forecaster](https://github.com/chenjj2/forecaster)
- Chen & Kipping (2017) [Probabilistic Forecasting of the Masses and Radii of Other Worlds](https://doi.org/10.3847/1538-4357/834/1/17)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## License

MIT License. See [`LICENSE`](LICENSE).