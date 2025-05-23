# ExoRM

- HomePage: https://github.com/kzhu2099/ExoRM
- Issues: https://github.com/kzhu2099/ExoRM/issues

[![PyPI Downloads](https://static.pepy.tech/badge/ExoRM)](https://pepy.tech/projects/ExoRM)

Author: Kevin Zhu

## Features

- continuous radius-mass relationship
- smooth with lower residuals
- simple usage, log10 and linear
- best-fit for Terran, Neptunian, and Jovian

## Installation

To install ExoRM, use pip: ```pip install ExoRM```.

However, many prefer to use a virtual environment.

macOS / Linux:

```sh
# make your desired directory
mkdir /path/to/your/directory
cd /path/to/your/directory

# setup the .venv (or whatever you want to name it)
pip install virtualenv
python3 -m venv .venv

# install ExoRM
source .venv/bin/activate
pip install ExoRM

deactivate # when you are completely done
```

Windows CMD:

```sh
# make your desired directory
mkdir C:path\to\your\directory
cd C:path\to\your\directory

# setup the .venv (or whatever you want to name it)
pip install virtualenv
python3 -m venv .venv

# install ExoRM
.venv\Scripts\activate
pip install ExoRM

deactivate # when you are completely done
```

## Usage

To first begin using ExoRM, the data and model must be initialized. This is due to the constant discovery of new exoplanets, adding to the data.

Furthermore, this requires periodic updating to include the most recent information.

Simply run `initialize_data()` and `initialize_model()`. Model initialization requires a smoothing amount, which is set to 280 but should be increased when there is more data. A plot of the model will be shown for you to see. Both are stored in your OS's Application Data for ExoRM. ExoRM provides built in functions to retrieve from this folder.

To use the model, call `ExoRM.load_model()` which returns the model from the filepath. If you wish, you may use `model.save(...)` to save it to your own directory.

The model supports log10 and linear scale in earth radii. When using the `model(), .__call__(), or .predict()`, the log10 scale is used. Linear predictions are used in `.predict_linear()`.

The high amount of uncertainty can be accessed from ExoRM. There is only log10 uncertainty due to the linear scale's differences, which may be accessed through `.calculate_error()` for the most recent values or `.error` for the value calculated at initialization.

ExoRM's data limitations required overrides for certain areas. By default, `override_min()` and `override_max()` are set to the inverse power law relationship found by Chen and kipping (2017). The transition points to those are smooth and are calculated to be the closest intersection between the model and the relationship.

An example is seen in the `example.ipynb`.

## License

The License is an MIT License found in the LICENSE file.