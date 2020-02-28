import math
from typing import *

from cognite.client._api.assets import AssetsAPI
from cognite.power.data_classes import PowerAsset, PowerAssetList
from cognite.power.exceptions import assert_single_result


class GenericPowerAPI(AssetsAPI):
    def __init__(self, metadata_filter, config, api_version, cognite_client):
        super().__init__(config, api_version, cognite_client)
        self.metadata_filter = metadata_filter

    def _all_bidding_areas(self):
        return self._cognite_client.assets.list(metadata={"type": "BiddingArea"})

    def list(
        self,
        grid_type: str = None,
        base_voltage: Iterable = None,
        bidding_areas: list = None,
        limit: int = None,
        **filters,
    ):
        if bidding_areas:
            subtree_assets = [a for a in self._all_bidding_areas() if a.name in bidding_areas]
            found_names = [a.name for a in subtree_assets]
            not_found = set(bidding_areas) - set(found_names)
            if not_found:
                raise ValueError(f"Bidding area(s) {not_found} not found")
            subtree_ids = [a.id for a in subtree_assets]
        else:
            subtree_ids = None
        filters["metadata"] = {**filters.get("metadata", {}), **self.metadata_filter}
        if grid_type:
            filters["metadata"]["Equipment.gridType"] = grid_type
        assets = super().list(asset_subtree_ids=subtree_ids, limit=limit, **filters)
        if base_voltage:
            assets = [
                a
                for a in assets
                if float((a.metadata or {}).get("BaseVoltage_nominalVoltage", math.nan)) in base_voltage
            ]
        return PowerAssetList._load_assets(assets, cognite_client=self._cognite_client)

    def retrieve_name(self, name):
        result = assert_single_result(super().list(name=name, metadata=self.metadata_filter))
        return PowerAsset._load_from_asset(result, self._cognite_client)
