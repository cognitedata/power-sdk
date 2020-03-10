
<a href="https://cognite.com/">
    <img src="https://github.com/cognitedata/cognite-python-docs/blob/master/img/cognite_logo.png" alt="Cognite logo" title="Cognite" align="right" height="80" />
</a>

Cognite Power SDK
=================
[![build](https://webhooks.dev.cognite.ai/build/buildStatus/icon?job=github-builds/cognite-power-sdk/master)](https://jenkins.cognite.ai/job/github-builds/job/cognite-power-sdk/job/master/)
[![Documentation Status](https://readthedocs.com/projects/cognite-power-sdk/badge/?version=latest)](https://cognite-docs.readthedocs-hosted.com/en/latest/)
[![PyPI version](https://badge.fury.io/py/cognite-sdk-experimental.svg)](https://pypi.org/project/cognite-experimental-sdk/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This is an extensions package to the [Cognite Python SDK](https://github.com/cognitedata/cognite-sdk-python)
 for developers testing features in development in Cognite Data Fusion (CDF). 

## Quickstart
Import a client with:

```python
from cognite.power import PowerClient
```
The resulting client object will contain all normal SDK functionality
in addition to Power SDK extensions.

## Documentation
* Power SDK Documentation is available by running `make html` in the `docs` directory.
* [SDK Documentation](https://cognite-docs.readthedocs-hosted.com/en/latest/)
* [API Documentation](https://doc.cognitedata.com/)
* [Cognite Developer Documentation](https://docs.cognite.com/dev/)

## Installation
To install this package:
```bash
$ pip install cognite-power-sdk
```

