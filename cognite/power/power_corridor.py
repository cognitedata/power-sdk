from typing import *

import pandas as pd

from cognite.client.data_classes._base import CogniteResourceList
from cognite.power.data_classes import PowerAsset, PowerAssetList


class PowerCorridorComponent:
    def __init__(self, asset: PowerAsset, sequence_number, fraction=1.0):
        self.asset = asset
        self.sequence_number = sequence_number
        self.fraction = fraction

    def dump(self, *args, **kwargs):
        return {
            "asset": f"{self.asset.__class__.__name__}: {self.asset.name}",
            "fraction": self.fraction,
            "sequence_number": self.sequence_number,
        }


class PowerCorridor(CogniteResourceList):
    _RESOURCE = PowerCorridorComponent
    _ASSERT_CLASSES = False

    def __init__(self, items: List[PowerCorridorComponent], cognite_client=None):
        super().__init__(items, cognite_client=cognite_client)
        self.assets = PowerAssetList([a.asset for a in items])
        assert self.assets.type  # exists/is homogeneous

    @property
    def fractions(self):
        return [pci.fraction for pci in self.data]

    def terminals(self):
        ts = [pci.asset.terminals(sequence_number=pci.sequence_number) for pci in self.data]
        unwrapped = []
        for t in ts:
            assert len(t) == 1
            unwrapped.append(t)
        return PowerAssetList(t)

    def time_series(self, measurement_type="ThreePhaseActivePower", timeseries_type="estimated_value"):
        ts = [t.time_series() for t in self.terminals()]
        unwrapped = []
        for t in ts:
            assert len(t) == 1
            unwrapped.append(t)
        return unwrapped

    def retrieve_dataframe(
        self,
        measurement_type="ThreePhaseActivePower",
        timeseries_type="estimated_value",
        thresholds: Dict = None,
        aggregates=None,
        **kwargs,
    ):
        ts = self.terminals().time_series(measurement_type=measurement_type, timeseries_type=timeseries_type)
        dfd = self._cognite_client.retrieve_dataframe_dict(aggregates=aggregates, **kwargs)
        ret = pd.DataFrame()
        for k, v in dfd.items():
            ret[k] = pd.sum(v * f for v, f in zip(v, self.fractions))
