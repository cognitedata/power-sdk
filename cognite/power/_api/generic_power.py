import math
from typing import *

from cognite.client._api.assets import AssetsAPI
from cognite.power.data_classes import *
from cognite.power.data_classes import _str_to_class
from cognite.power.exceptions import assert_single_result


class GenericPowerAPI(AssetsAPI):
    def __init__(self, power_type, config, api_version, cognite_client, grid_type_field="Equipment.gridType"):
        self.power_type = power_type
        self.asset_class = _str_to_class(self.power_type)
        self.grid_type_field_name = grid_type_field
        super().__init__(config, api_version, cognite_client)
        self.metadata_filter = {"type": power_type}

    def _all_bidding_areas(self):
        return self._cognite_client.assets.list(metadata={"type": "BiddingArea"})

    def list(
        self,
        grid_type: str = None,
        base_voltage: Iterable = None,
        bidding_areas: list = None,
        limit: int = None,
        **filters,
    ) -> PowerAssetList:
        """Lists power assets. Supports all parameters as the normal list function in addition to some power specific ones.

        Args:
            grid_type: filters on Equipment.gridType
            base_voltage: filters on BaseVoltage_nominalVoltage in the given range or list.
            bidding_areas: filters on assets being in the bidding areas with this name.
            filters: all other parameters for the normal AssetsAPI.list method
        """
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
            filters["metadata"][self.grid_type_field_name] = grid_type
        assets = super().list(asset_subtree_ids=subtree_ids, limit=limit, **filters)
        if base_voltage:
            assets = [
                a
                for a in assets
                if float((a.metadata or {}).get("BaseVoltage_nominalVoltage", math.nan)) in base_voltage
            ]
        return PowerAssetList._load_assets(assets, self.power_type, cognite_client=self._cognite_client)

    def retrieve_name(self, name: str) -> PowerAsset:
        """Retrieve a single asset by exact name match. Fails if not exactly one asset is found."""
        result = assert_single_result(super().list(name=name, metadata=self.metadata_filter))
        return PowerAsset._load_from_asset(result, self.power_type, self._cognite_client)
