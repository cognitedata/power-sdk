import functools
import math
from typing import *

from cognite.client._api.assets import AssetsAPI
from cognite.power.data_classes import PowerAsset, PowerAssetList, _str_to_class
from cognite.power.exceptions import assert_single_result


class GenericPowerAPI(AssetsAPI):
    def __init__(self, power_type, config, api_version, cognite_client, grid_type_field="Equipment.gridType"):
        self.power_type = power_type
        if power_type:
            assert _str_to_class(self.power_type) is not None
        self.grid_type_field_name = grid_type_field
        super().__init__(config, api_version, cognite_client)

    @functools.lru_cache(maxsize=1)
    def _all_bidding_area(self):
        return self._cognite_client.assets.list(metadata={"type": "BiddingArea"})

    def _create_filter(
        self,
        grid_type: str = None,
        bidding_area: Union[str, List[str]] = None,
        asset_type: Union[str, List[str]] = None,
        wrap_ids=False,
        **filters,
    ) -> Dict:

        if bidding_area:
            if isinstance(bidding_area, str):
                bidding_area = [bidding_area]
            upcased_bidding_area = [name.upper() for name in bidding_area]
            subtree_assets = [a for a in self._all_bidding_area() if a.name.upper() in upcased_bidding_area]
            found_names = [a.name.upper() for a in subtree_assets]
            not_found = set(upcased_bidding_area) - set(found_names)
            if not_found:
                raise ValueError(
                    f"Bidding area(s) {not_found} not found - should be one of {[a.name.upper() for a in self._all_bidding_area()]} (case insensitive)"
                )
            if wrap_ids:
                filters["asset_subtree_ids"] = [{"id": a.id} for a in subtree_assets]
            else:
                filters["asset_subtree_ids"] = [a.id for a in subtree_assets]
        if "metadata" not in filters:
            filters["metadata"] = {}
        asset_type = asset_type or self.power_type
        if asset_type:
            filters["metadata"]["type"] = asset_type
        if grid_type:
            if not grid_type.startswith("GridTypeKind."):
                grid_type = "GridTypeKind." + grid_type
            filters["metadata"][self.grid_type_field_name] = grid_type
        return filters

    def list(
        self,
        grid_type: str = None,
        base_voltage: Iterable = None,
        bidding_area: Union[str, List[str]] = None,
        asset_type: Union[str, List[str]] = None,
        limit: int = None,
        **kwargs,
    ) -> PowerAssetList:
        """Lists power assets. Supports all parameters as the normal list function in addition to some power specific ones.

        Args:
            grid_type (str): filters on Equipment.gridType. Can give "GridTypeKind.regional" or just "regional" etc.
            base_voltage (Iterable): filters on BaseVoltage_nominalVoltage in the given range or list.
            bidding_area (Union[str, List[str]]): filters on assets being in the bidding areas with this (case-insensitive) name.
            asset_type (Union[str, List[str]]): filter on these asset types. Automatically populated for specific APIs
            kwargs: all other parameters for the normal AssetsAPI.list method

        Returns:
            PowerAssetList: List of the requested assets.
        """
        if (base_voltage is not None or isinstance(asset_type, list)) and limit not in [None, -1, float("inf")]:
            raise ValueError("Can not set a limit when specifying a base voltage filter or multiple asset types")
        if self.power_type and asset_type:
            raise ValueError("Can not filter on asset_types in this API, use client.power_assets instead")
        if isinstance(asset_type, list):
            asset_lists = [
                self.list(
                    asset_type=type, bidding_area=bidding_area, base_voltage=base_voltage, grid_type=grid_type, **kwargs
                )
                for type in asset_type
            ]
            return PowerAssetList._load_assets(
                sum(asset_lists, []),
                asset_type[0] if len(asset_type) == 1 else None,  # mixed
                base_voltage=base_voltage,
                cognite_client=self._cognite_client,
            )
        filters = self._create_filter(bidding_area=bidding_area, grid_type=grid_type, asset_type=asset_type, **kwargs)
        assets = super().list(limit=limit, **filters)
        return PowerAssetList._load_assets(
            assets,
            filters.get("metadata", {}).get("type"),
            base_voltage=base_voltage,
            cognite_client=self._cognite_client,
        )

    def search(
        self,
        name: str = None,
        grid_type: str = None,
        base_voltage: Iterable = None,
        bidding_area: Union[str, List[str]] = None,
        asset_type: Union[str, List[str]] = None,
        limit: int = None,
        **kwargs,
    ) -> PowerAssetList:
        """Search power assets. Supports all parameters as the normal search function in addition to some power specific ones.

        Args:
            name (str): Fuzzy search on name.
            grid_type (str): filters on Equipment.gridType. Can give "GridTypeKind.regional" or just "regional" etc.
            base_voltage (Iterable): filters on BaseVoltage_nominalVoltage in the given range or list.
            bidding_area (Union[str, List[str]]): filters on assets being in the bidding areas with this name.
            asset_type (Union[str, List[str]]): filter on these asset types. Automatically populated for specific APIs
            kwargs: all other parameters for the normal AssetsAPI.list method

        Returns:
            PowerAssetList: List of the requested assets.
        """
        if self.power_type and asset_type:
            raise ValueError("Can not filter on asset_types in this API, use client.power_assets instead")
        if isinstance(asset_type, list):
            asset_lists = [
                self.search(
                    name=name,
                    asset_type=type,
                    bidding_area=bidding_area,
                    base_voltage=base_voltage,
                    grid_type=grid_type,
                    **kwargs,
                )
                for type in asset_type
            ]
            return PowerAssetList._load_assets(
                sum(asset_lists, []),
                asset_type[0] if len(asset_type) == 1 else None,  # mixed
                base_voltage=base_voltage,
                cognite_client=self._cognite_client,
            )

        filter = self._create_filter(
            bidding_area=bidding_area,
            grid_type=grid_type,
            asset_type=asset_type,
            wrap_ids=True,
            **kwargs.get("filter", {}),
        )
        if "filter" in kwargs:
            del kwargs["filter"]

        assets = super().search(name=name, limit=limit, filter=filter, **kwargs)
        return PowerAssetList._load_assets(
            assets,
            filter.get("metadata", {}).get("type"),
            base_voltage=base_voltage,
            cognite_client=self._cognite_client,
        )

    def retrieve_name(
        self, name: Union[List[str], str], asset_type: str = None, bidding_area: Union[str, List[str]] = None
    ) -> Union[PowerAsset, PowerAssetList]:
        """Retrieve one or more assets by exact name match. Fails if not exactly one asset is found.

        Args:
            name (Union[List[str], str]): One or more names to search for.
            asset_type (Union[str, List[str]]): filter on these asset types. Automatically populated for specific APIs
            bidding_area (Union[str, List[str]]): filters on assets being in the bidding areas with this name.

        Returns:
            Union[PowerAsset,PowerAssetList]: The requested asset(s).
        """
        if isinstance(name, str):
            filters = self._create_filter(asset_type=asset_type, bidding_area=bidding_area)
            result = assert_single_result(super().list(name=name, **filters))
            return PowerAsset._load_from_asset(result, self.power_type or asset_type, self._cognite_client)
        else:
            return PowerAssetList(
                [self.retrieve_name(name=n, asset_type=asset_type, bidding_area=bidding_area) for n in name],
                cognite_client=self._cognite_client,
            )

    def retrieve(self, *args, **kwargs) -> Optional[PowerAsset]:
        asset = super().retrieve(*args, **kwargs)
        if asset:
            return PowerAsset._load_from_asset(asset, self.power_type, self._cognite_client)

    def retrieve_multiple(self, *args, **kwargs) -> PowerAssetList:
        return PowerAssetList._load_assets(
            super().retrieve_multiple(*args, **kwargs), self.power_type, self._cognite_client
        )
