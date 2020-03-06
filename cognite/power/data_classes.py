import math
import sys
import warnings
from typing import *

import numpy as np

from cognite.client.data_classes import Asset, AssetList, AssetUpdate, TimeSeriesList
from cognite.client.utils._concurrency import execute_tasks_concurrently
from cognite.power.exceptions import WrongPowerTypeError, assert_single_result


def _str_to_class(classname):
    if not classname:
        return None
    return getattr(sys.modules[__name__], classname, None)


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
    def base_voltage(self) -> Optional[float]:
        bv = self.metadata_value("BaseVoltage_nominalVoltage")
        if bv is not None:
            return float(bv)
        else:
            return None

    def relationship_sources(
        self,
        power_type,
        resource_type="Asset",
        relationship_type="belongsTo",
        base_voltage: Iterable = None,
        x_filter: Callable = None,
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
            self._cognite_client, [r.source["resourceId"] for r in rels], power_type, base_voltage, x_filter
        )

    def relationship_targets(
        self,
        power_type,
        resource_type="Asset",
        relationship_type="belongsTo",
        base_voltage: Iterable = None,
        x_filter: Callable = None,
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
            self._cognite_client, [r.target["resourceId"] for r in rels], power_type, base_voltage, x_filter
        )

    @staticmethod
    def _filter_and_convert(
        client, external_ids, power_type, base_voltage: Iterable = None, x_filter: Callable = None
    ) -> "PowerAssetList":
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
        return PowerAssetList._load_assets(
            assets, power_type, cognite_client=client, base_voltage=base_voltage, x_filter=x_filter
        )

    def analogs(self):
        """Shortcut for finding the associated Analogs (via it's terminals)"""
        return self.terminals().analogs()

    def time_series(self, measurement_type=None, timeseries_type=None, **kwargs):
        metadata_filter = {"measurement_type": measurement_type, "timeseries_type": timeseries_type, **kwargs}
        metadata_filter = {k: v for k, v in metadata_filter.items() if v}
        return self._cognite_client.time_series.list(asset_subtree_ids=[self.id], metadata=metadata_filter)

    @staticmethod
    def _load_from_asset(asset, class_name, cognite_client):
        cls = _str_to_class(class_name) or PowerAsset
        return cls(cognite_client=cognite_client, **asset.dump())


class Terminal(PowerAsset):
    def analogs(self):
        """Shortcut for finding the associated Analogs for a Terminal"""
        return self.relationship_sources("Analog")

    @property
    def sequence_number(self) -> Optional[int]:
        seq_number = self.metadata_value("Terminal.sequenceNumber")
        if seq_number is not None:
            return int(seq_number)
        else:
            return None


class Analog(PowerAsset):
    pass


class PowerTransformer(PowerAsset):
    def ac_line_segments(self, base_voltage: Iterable = None):
        return self.relationship_targets("ACLineSegment", relationship_type="connectsTo", base_voltage=base_voltage)

    def power_transformer_ends(self, end_number: Optional[Union[int, Iterable]] = None, base_voltage: Iterable = None):
        """Shortcut for finding the associated PowerTransformerEnds for a PowerTransformer
        Args:
            end_number: filter on transformer end number
        """
        if end_number is not None:
            if not isinstance(end_number, Iterable):
                end_number = [end_number]
            end_number_filter = lambda a: a.end_number in end_number
        else:
            end_number_filter = None
        return self.relationship_sources("PowerTransformerEnd", base_voltage=base_voltage, x_filter=end_number_filter)

    def terminals(self, terminal_number: Optional[Union[int, Iterable]] = None):
        return self.power_transformer_ends().terminals(terminal_number=terminal_number)

    def substation(self):
        return assert_single_result(self.relationship_targets("Substation"))


class Substation(PowerAsset):
    def power_transformers(self):
        return self.relationship_sources("PowerTransformer")

    def terminals(self):
        """Shortcut for finding the associated Terminals"""
        return self.relationship_sources("Terminal", relationship_type="connectsTo")

    def ac_line_segments(self, base_voltage: Iterable = None):
        return self.terminals().ac_line_segments(base_voltage=base_voltage)


class PowerTransformerEnd(PowerAsset):
    def terminals(self, terminal_number: Optional[Union[int, Iterable]]):
        """Shortcut for finding the associated Terminals"""
        if terminal_number is not None:
            if not isinstance(terminal_number, Iterable):
                terminal_number = [terminal_number]
            terminal_number_filter = lambda a: int(a.metadata.get("Terminal.sequenceNumber", -1)) in terminal_number
        else:
            terminal_number_filter = None
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=terminal_number_filter)

    def substation(self):
        return self.power_transformer().substation()

    def power_transformer(self):
        return assert_single_result(self.relationship_targets("PowerTransformer"))

    @property
    def end_number(self) -> Optional[int]:
        end_number = self.metadata_value("TransformerEnd.endNumber")
        if end_number is not None:
            return int(end_number)
        else:
            return None

    def opposite_end(self):
        end_number = self.end_number
        if end_number not in [1, 2]:
            raise ValueError(f"Can't get opposite end for list with end number {end_number}, should be all 1 or all 2")
        opposite_end_number = 1 if end_number == 2 else 2
        return assert_single_result(self.power_transformer().power_transformer_ends(end_number=opposite_end_number))


class GeneratingUnit(PowerAsset):
    pass


class WindGeneratingUnit(GeneratingUnit):
    pass


class HydroGeneratingUnit(GeneratingUnit):
    pass


class SynchronousMachine(PowerAsset):
    def terminals(self):
        """Shortcut for finding the associated Terminals"""
        return self.relationship_sources("Terminal", relationship_type="connectsTo")

    def GeneratingUnit(self) -> GeneratingUnit:
        if self.type != "SynchronousMachine":
            raise WrongPowerTypeError(
                "Can only find the power GeneratingUnit for a SynchronousMachine, not for a  {}.".format(self.type)
            )
        return PowerAsset._load_from_asset(
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

    def substations(self):
        return self.terminals().substations()


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    @property
    def type(self) -> Optional[str]:
        if not self.data:
            return None
        types = list({a.type for a in self.data})
        if len(types) != 1:
            raise Exception(f"Can not determine type of list with assets of types {', '.join(types)}")
        return types[0]

    def has_type(self, check_type):
        return self.type == check_type

    @staticmethod
    def _load_assets(assets, class_name, cognite_client, base_voltage: Iterable = None, x_filter: Callable = None):
        power_assets = [PowerAsset._load_from_asset(a, class_name, cognite_client) for a in assets]
        if x_filter:
            power_assets = [a for a in power_assets if x_filter(a)]
        if base_voltage is not None:
            power_assets = [a for a in power_assets if a.base_voltage in base_voltage]
        return PowerAssetList(power_assets, cognite_client=cognite_client)

    def relationship_sources(
        self,
        power_type,
        resource_type="Asset",
        relationship_type="belongsTo",
        base_voltage: Iterable = None,
        x_filter: Callable = None,
    ) -> "PowerAssetList":
        """Shortcut for finding all assets that are a source, with the current assets as a target"""
        if not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        rels = []
        for si in range(0, len(self.data), 1000):
            rels += self._cognite_client.relationships.list(
                source_resource=resource_type,
                targets=[{"resource": "Asset", "resourceId": a.external_id} for a in self.data[si : si + 1000]],
                relationship_type=relationship_type,
                limit=None,
            )
        return PowerAsset._filter_and_convert(
            self._cognite_client,
            [r.source["resourceId"] for r in rels],
            power_type,
            base_voltage=base_voltage,
            x_filter=x_filter,
        )

    def relationship_targets(
        self,
        power_type,
        resource_type="Asset",
        relationship_type="belongsTo",
        base_voltage: Iterable = None,
        x_filter: Callable = None,
    ) -> "PowerAssetList":
        """Shortcut for finding all assets that are a target, with the current assets as a source"""
        if not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        rels = []
        for si in range(0, len(self.data), 1000):
            rels += self._cognite_client.relationships.list(
                target_resource=resource_type,
                sources=[{"resource": "Asset", "resourceId": a.external_id} for a in self.data[si : si + 1000]],
                relationship_type=relationship_type,
                limit=None,
            )
        return PowerAsset._filter_and_convert(
            self._cognite_client,
            [r.target["resourceId"] for r in rels],
            power_type,
            base_voltage=base_voltage,
            x_filter=x_filter,
        )

    def power_transformer_ends(self, end_number: Optional[Union[int, Iterable]] = None, base_voltage: Iterable = None):
        """Shortcut for finding the associated PowerTransformerEnds for a list of PowerTransformers
        Args:
            end_number: filter on transformer end number
        """
        if end_number is not None:
            if not isinstance(end_number, Iterable):
                end_number = [end_number]
            end_number_filter = lambda a: a.end_number in end_number
        else:
            end_number_filter = None
        if self.has_type("Terminal"):
            return self.relationship_targets(
                "PowerTransformerEnd",
                base_voltage=base_voltage,
                x_filter=end_number_filter,
                relationship_type="connectsTo",
            )
        else:
            return self.relationship_sources(
                "PowerTransformerEnd", base_voltage=base_voltage, x_filter=end_number_filter
            )

    def power_transformers(self):
        if self.has_type("PowerTransformerEnd"):
            return self.relationship_targets("PowerTransformer")
        elif self.has_type("Substation"):
            return self.relationship_sources("PowerTransformer")
        elif not self.data:
            return PowerAssetList([])
        else:
            raise ValueError(f"Can't get substations for a list of {self.type}")

    def substations(self):
        if self.has_type("PowerTransformer"):
            return self.relationship_targets("Substation")
        elif self.has_type("Terminal"):
            return self.relationship_targets("Substation", relationship_type="connectsTo")
        elif self.has_type("ACLineSegment"):
            return self.terminals().substations()
        elif not self.data:
            return PowerAssetList([])
        else:
            raise ValueError(f"Can't get substations for a list of {self.type}")

    def ac_line_segments(self, base_voltage: Iterable = None):
        if self.has_type("PowerTransformer"):
            return self.substations().ac_line_segments(base_voltage=base_voltage)
        if self.has_type("Substation"):
            return self.terminals().ac_line_segments(base_voltage=base_voltage)
        elif self.has_type("Terminal"):
            return self.relationship_targets("ACLineSegment", relationship_type="connectsTo", base_voltage=base_voltage)
        elif not self.data:
            return PowerAssetList([])
        else:
            raise ValueError(f"Can't get substations for a list of {self.type}")

    def terminals(self, terminal_number: Optional[Union[int, Iterable]] = None):
        """Shortcut for finding the associated Terminals. For a power transformer list, will retrieve all terminals via terminal ends of the specified end_number(s)"""

        if self.has_type("PowerTransformer"):
            return self.power_transformer_ends().terminals(terminal_number=terminal_number)

        if terminal_number is not None:
            if not isinstance(terminal_number, Iterable):
                terminal_number = [terminal_number]
            terminal_number_filter = lambda a: int(a.metadata.get("Terminal.sequenceNumber", -1)) in terminal_number
        else:
            terminal_number_filter = None
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=terminal_number_filter)

    def analogs(self):
        """Shortcut for finding the associated Analogs. Only works when applied directly to a list of Terminals."""
        return self.relationship_sources("Analog")

    def opposite_ends(self):
        if self.has_type("PowerTransformerEnd"):
            end_numbers = {a.end_number for a in self.data}
            if end_numbers != {1} and end_numbers != {2}:
                raise ValueError(
                    f"Can't get opposite end for list with end number(s) {end_numbers}, should be either all 1 or all 2"
                )
            opposite_end_number = 1 if list(end_numbers)[0] == 2 else 2
            return self.power_transformers().power_transformer_ends(end_number=opposite_end_number)
        else:
            raise ValueError(f"Can't get substations for a list of {self.type}")

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
