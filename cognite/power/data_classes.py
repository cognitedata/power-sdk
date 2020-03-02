import sys
import warnings
from typing import *

import numpy as np

from cognite.client.data_classes import Asset, AssetList, AssetUpdate, TimeSeriesList
from cognite.client.utils._concurrency import execute_tasks_concurrently
from cognite.power.exceptions import WrongPowerTypeError, assert_single_result


def _str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class PowerAsset(Asset):
    """Extended asset class for Power related resources"""

    def metadata_value(self, key: str) -> Optional[str]:
        "Safe way of getting a metadata value, returns None if metadata or key does not exist"
        return (self.metadata or {}).get(key)

    @property
    def type(self):
        """Shortcut for it's metadata-defined type (e.g. 'TransformerEnd')"""
        return self.metadata_value("type")

    @property
    def location(self):
        metadata = self.metadata or {}
        x, y = metadata.get("Location X"), metadata.get("Location Y")
        return (float(x) if x else None, float(y) if y else None)

    def dump(self, camel_case: bool = False, expand=()) -> Dict[str, Any]:
        dump = super().dump(camel_case)
        metadata = self.metadata or {}
        for field in expand:
            value = metadata.get(field)
            if value is not None:
                dump[field] = value
        lx, ly = self.location
        dump["Location X"] = lx
        dump["Location Y"] = ly
        return dump

    def relationship_sources(
        self, power_type, resource_type="Asset", relationship_type="belongsTo", x_filter: Callable = None
    ):
        """Shortcut for finding all assets that are a source, with the current asset as a target"""
        rels = self._cognite_client.relationships.list(
            source_resource=resource_type,
            target_resource="Asset",
            target_resource_id=self.external_id,
            relationship_type=relationship_type,
            limit=None,
        )
        return PowerAsset._filter_and_convert(
            self._cognite_client, [r.source["resourceId"] for r in rels], power_type, x_filter
        )

    def relationship_targets(
        self, power_type, resource_type="Asset", relationship_type="belongsTo", x_filter: Callable = None
    ):
        """Shortcut for finding all assets that are a target, with the current asset as a source"""
        rels = self._cognite_client.relationships.list(
            target_resource=resource_type,
            source_resource="Asset",
            source_resource_id=self.external_id,
            relationship_type=relationship_type,
            limit=None,
        )
        return PowerAsset._filter_and_convert(
            self._cognite_client, [r.target["resourceId"] for r in rels], power_type, x_filter
        )

    @staticmethod
    def _filter_and_convert(client, external_ids, power_type, x_filter: Callable = None) -> "PowerAssetList":
        external_ids = list(np.unique(external_ids))
        assets = client.assets.retrieve_multiple(external_ids=external_ids, ignore_unknown_ids=True)
        if len(assets) != len(external_ids):
            warnings.warn(
                "{} assets not found when looking up {}s among {}".format(
                    len(external_ids) - len(assets), power_type, external_ids
                )
            )
        if power_type:
            assets = [a for a in assets if (a.metadata or {}).get("type") == power_type]
        if x_filter:
            assets = [a for a in assets if x_filter(a)]
        return PowerAssetList._load_assets(assets, power_type, cognite_client=client)

    def analogs(self):
        """Shortcut for finding the associated Analogs (via it's terminals)"""
        return self.terminals().analogs()

    def time_series(self, measurement_type=None, timeseries_type=None, **kwargs):
        metadata_filter = {"measurement_type": measurement_type, "timeseries_type": timeseries_type, **kwargs}
        metadata_filter = {k: v for k, v in metadata_filter.items() if v}
        return self._cognite_client.time_series.list(asset_subtree_ids=[self.id], metadata=metadata_filter)

    @staticmethod
    def _load_from_asset(asset, class_name, cognite_client):
        return _str_to_class(class_name)(cognite_client=cognite_client, **asset.dump())


class Terminal(PowerAsset):
    def analogs(self):
        """Shortcut for finding the associated Analogs for a Terminal"""
        return self.relationship_sources("Analog")


class Analog(PowerAsset):
    pass


class PowerTransformer(PowerAsset):
    def ac_line_segments(self):
        return self.relationship_targets("ACLineSegment", relationship_type="connectsTo")

    def power_transformer_ends(self, end_number: Optional[int] = None):
        """Shortcut for finding the associated PowerTransformerEnds for a PowerTransformer
        Args:
            end_number: filter on transformer end number
        """
        if end_number is not None:
            end_number_filter = lambda a: int(a.metadata.get("TransformerEnd.endNumber", -1)) == end_number
        else:
            end_number_filter = None
        return self.relationship_sources("PowerTransformerEnd", x_filter=end_number_filter)

    def terminals(self):
        return self.power_transformer_ends().terminals()

    def substation(self):
        return assert_single_result(self.relationship_targets("Substation"))


class Substation(PowerAsset):
    def power_transformers(self):
        if self.type != "Substation":
            raise WrongPowerTypeError(
                "Can only find the transformers for a substation, not for a  {}.".format(self.type)
            )
        return self.relationship_sources("PowerTransformer")

    def terminals(self):
        """Shortcut for finding the associated Terminals"""
        return self.relationship_sources("Terminal")


class PowerTransformerEnd(PowerAsset):
    def terminals(self):
        """Shortcut for finding the associated Terminals"""
        return self.relationship_sources("Terminal")

    def substation(self):
        return assert_single_result(self.relationship_targets("PowerTransformer")).substation()


class GeneratingUnit(PowerAsset):
    pass


class WindGeneratingUnit(GeneratingUnit):
    pass


class HydroGeneratingUnit(GeneratingUnit):
    pass


class SynchronousMachine(PowerAsset):
    def terminals(self):
        """Shortcut for finding the associated Terminals"""
        return self.relationship_sources("Terminal")

    def GeneratingUnit(self) -> GeneratingUnit:
        if self.type != "SynchronousMachine":
            raise WrongPowerTypeError(
                "Can only find the power GeneratingUnit for a SynchronousMachine, not for a  {}.".format(self.type)
            )
        return PowerAsset._load(
            assert_single_result([a for a in self.relationship_sources() if "GeneratingUnit" in a.type]),
            "GeneratingUnit",
            self._cognite_client,
        )

    def substation(self):
        """Shortcut for finding the associated transformer for a PowerTransformer, PowerTransformerEnd, .."""
        return self.GeneratingUnit().substation()


class ACLineSegment(PowerAsset):
    def terminals(self):
        """Shortcut for finding the connected Terminals for an ACLineSegment"""
        return self.relationship_sources("Terminal", relationship_type="connectsTo")


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    def dump(self, camel_case: bool = False) -> List[Dict[str, Any]]:
        return [resource.dump(camel_case, expand=("type",)) for resource in self.data]

    @staticmethod
    def _load_assets(assets, class_name, cognite_client):
        return PowerAssetList(
            [PowerAsset._load_from_asset(a, class_name, cognite_client) for a in assets], cognite_client=cognite_client
        )

    # TODO: chunk if list size > 1000
    def relationship_sources(
        self, power_type, resource_type="Asset", relationship_type="belongsTo", x_filter: Callable = None
    ) -> "PowerAssetList":
        """Shortcut for finding all assets that are a source, with the current assets as a target"""
        rels = []
        for si in range(0, len(self.data), 1000):
            rels += self._cognite_client.relationships.list(
                source_resource=resource_type,
                targets=[{"resource": "Asset", "resourceId": a.external_id} for a in self.data[si : si + 1000]],
                relationship_type=relationship_type,
                limit=None,
            )
        return PowerAsset._filter_and_convert(
            self._cognite_client, [r.source["resourceId"] for r in rels], power_type, x_filter
        )

    def relationship_targets(
        self, power_type, resource_type="Asset", relationship_type="belongsTo", x_filter: Callable = None
    ) -> "PowerAssetList":
        """Shortcut for finding all assets that are a target, with the current assets as a source"""
        rels = []
        for si in range(0, len(self.data), 1000):
            rels += self._cognite_client.relationships.list(
                target_resource=resource_type,
                sources=[{"resource": "Asset", "resourceId": a.external_id} for a in self.data[si : si + 1000]],
                relationship_type=relationship_type,
                limit=None,
            )
        return PowerAsset._filter_and_convert(
            self._cognite_client, [r.target["resourceId"] for r in rels], power_type, x_filter
        )

    def power_transformer_ends(self, end_number: Optional[int] = None):
        """Shortcut for finding the associated PowerTransformerEnds for a list of PowerTransformers
        Args:
            end_number: filter on transformer end number
        """
        if end_number is not None:
            end_number_filter = lambda a: int(a.metadata.get("TransformerEnd.endNumber", -1)) == end_number
        else:
            end_number_filter = None
        return self.relationship_sources("PowerTransformerEnd", x_filter=end_number_filter)

    def terminals(self):
        """Shortcut for finding the associated Terminals"""
        relationship_type = "connectsTo" if self.data[0].type == "ACLineSegment" else "belongsTo"
        return self.relationship_sources("Terminal", relationship_type=relationship_type)

    def analogs(self):
        """Shortcut for finding the associated Analogs. Only works when applied directly to a list of Terminals."""
        return self.relationship_sources("Analog")

    def time_series(self, measurement_type=None, timeseries_type=None, **kwargs):
        metadata_filter = {"measurement_type": measurement_type, "timeseries_type": timeseries_type, **kwargs}
        metadata_filter = {k: v for k, v in metadata_filter.items() if v}
        chunk_size = 100
        ids = [a.id for a in self.data]
        tasks = [
            {"asset_subtree_ids": ids[i : i + self._retrieve_chunk_size], "metadata": metadata_filter}
            for i in range(0, len(ids), chunk_size)
        ]
        res_list = execute_tasks_concurrently(self._cognite_client.time_series.list, tasks, max_workers=10)
        return TimeSeriesList(sum(res_list.joined_results(), []))
