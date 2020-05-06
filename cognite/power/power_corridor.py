from collections import UserList
from datetime import datetime
from typing import *

import numpy as np
import pandas as pd

from cognite.client.data_classes._base import CogniteResourceList
from cognite.client.utils._time import granularity_to_ms
from cognite.power.data_classes import PowerAsset, PowerAssetList
from cognite.power.exceptions import assert_single_result


class PowerCorridorComponent:
    def __init__(
        self,
        asset: PowerAsset,
        sequence_number,
        fraction=1.0,
        measurement_type="ThreePhaseActivePower",
        timeseries_type="estimated_value",
    ):
        self.asset = asset
        self.sequence_number = sequence_number
        self.fraction = fraction
        self.measurement_type = measurement_type
        self.timeseries_type = timeseries_type

    def dump(self, *args, **kwargs):
        return {
            "asset": f"{self.asset.__class__.__name__}: {self.asset.name}",
            "sequence_number": self.sequence_number,
            "fraction": self.fraction,
            "measurement_type": self.measurement_type,
            "timeseries_type": self.timeseries_type,
        }

    def terminal(self):
        term = [self.asset.terminals(sequence_number=self.sequence_number)]
        return assert_single_result(
            term,
            f"Expected a single terminal with sequence number {self.sequence_number} for {self.asset.name}, but found {len(term)}",
        )

    def time_series(self):
        term = self.terminal().time_series(measurement_type=self.measurement_type, timeseries_type=self.timeseries_type)
        return assert_single_result(
            term,
            f"Expected a single time series with measurement type {self.measurement_type} and timeseries type {self.timeseries_type} for {self.asset.name}, but found {len(term)}",
        )


class PowerCorridor(CogniteResourceList):
    _RESOURCE = PowerCorridorComponent
    _ASSERT_CLASSES = False

    def __init__(self, items: List[PowerCorridorComponent], cognite_client=None):
        """Power Corridor class. When cognite_client is ommitted, it is taken from the first asset given."""
        if not cognite_client and items:
            cognite_client = items[0].asset._cognite_client
        super().__init__(items, cognite_client=cognite_client)
        self.assets = PowerAssetList([a.asset for a in items])

    @property
    def fractions(self) -> List[float]:
        return [pci.fraction for pci in self.data]

    def calculate(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime] = "now",
        aggregates: Iterable = ("average", "max"),
        granularity: str = "10m",
        thresholds: Dict[str, float] = None,
    ) -> "pd.DataFrame":
        """Calculates a dataframe for a PowerCorridor. The results are approximated as sums of aggregates."""
        ts = [pci.time_series() for pci in self.data]
        dfd = self._cognite_client.datapoints.retrieve_dataframe_dict(
            id=[t.id for t in ts], start=start, end=end, granularity=granularity, aggregates=list(aggregates)
        )
        snitt = pd.DataFrame()
        for k, v in dfd.items():
            scaled_ts = [col * frac for (colname, col), frac in zip(dfd[k].items(), self.fractions)]
            snitt[k] = pd.concat(scaled_ts, axis=1).dropna().sum(axis=1)
        snitt = snitt.reindex(
            np.arange(
                snitt.index[0],
                snitt.index[-1] + pd.Timedelta(microseconds=1),
                pd.Timedelta(microseconds=granularity_to_ms(granularity) * 1000),
            ),
            copy=False,
        )
        for k, v in (thresholds or {}).items():
            snitt[k] = v
        return snitt
