from typing import *

from cognite.client._api.assets import AssetsAPI
from cognite.power.data_classes import PowerAsset, PowerAssetList
from cognite.power.exceptions import assert_single_result


class GenericPowerAPI(AssetsAPI):
    def __init__(self, metadata_filter, config, api_version, cognite_client):
        super().__init__(config, api_version, cognite_client)
        self.metadata_filter = metadata_filter

    def list(self, grid_type: str = None, base_voltage: Iterable = None, limit: int = None, **filters):
        filters["metadata"] = {**filters.get("metadata", {}), **self.metadata_filter}
        if grid_type:
            filters["metadata"]["Equipment.gridType"] = grid_type
        assets = super().list(limit=limit, **filters)
        if base_voltage:
            assets = [a for a in assets if int((a.metadata or {}).get("BaseVoltage")) in base_voltage]
        return PowerAssetList._load_assets(assets, cognite_client=self._cognite_client)

    def retrieve_name(self, name):
        result = assert_single_result(super().list(name=name, metadata=self.metadata_filter))
        return PowerAsset._load_from_asset(result, self._cognite_client)
