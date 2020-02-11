import pytest

from cognite.power import PowerClient


def test_client():
    client = PowerClient(api_key="??", project="test")
