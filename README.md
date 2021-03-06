
<a href="https://cognite.com/">
    <img src="https://github.com/cognitedata/cognite-python-docs/blob/master/img/cognite_logo.png" alt="Cognite logo" title="Cognite" align="right" height="80" />
</a>

Cognite Power SDK
=================
[![Release Status](https://github.com/cognitedata/power-sdk/workflows/release/badge.svg)](https://github.com/cognitedata/power-sdk/actions)
[![Documentation Status](https://readthedocs.com/projects/cognite-power-sdk/badge/?version=latest)](https://cognite-power-sdk.readthedocs-hosted.com/en/latest/)
[![PyPI version](https://badge.fury.io/py/cognite-power-sdk.svg)](https://pypi.org/project/cognite-power-sdk/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


This is an extensions package to the [Cognite Python SDK](https://github.com/cognitedata/cognite-sdk-python)
 for power grid specific applications in Cognite Data Fusion (CDF). 

## Quickstart
Import a client with:

```python
from cognite.power import PowerClient
```
The resulting client object will contain all normal SDK functionality
in addition to Power SDK extensions.

## Documentation
* [Power SDK Documentation](https://cognite-power-sdk.readthedocs-hosted.com/en/latest/)
* [SDK Documentation](https://cognite-docs.readthedocs-hosted.com/en/latest/)
* [API Documentation](https://doc.cognitedata.com/)
* [Cognite Developer Documentation](https://docs.cognite.com/dev/)

## Installation
To install this package:
```bash
$ pip install cognite-power-sdk
```
For PowerGraph plotting support in Jupyter lab, install the relevant extensions using
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install jupyterlab-plotly --no-build
jupyter labextension install plotlywidget --no-build
jupyter lab build
```
