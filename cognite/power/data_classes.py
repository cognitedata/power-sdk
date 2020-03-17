import math
import sys
import warnings
from collections import defaultdict
from typing import *

import numpy as np

from cognite.client.data_classes import Asset, AssetList, AssetUpdate, TimeSeriesList
from cognite.client.utils._concurrency import execute_tasks_concurrently
from cognite.power.exceptions import (
    MixedPowerAssetListException,
    SinglePowerAssetExpected,
    WrongPowerTypeError,
    assert_single_result,
)


def _str_to_class(classname):
    if not classname:
        return None
    return getattr(sys.modules[__name__], classname, None)


class PowerAsset(Asset):
    """Extended asset class for Power related resources"""

    def metadata_value(self, key: str) -> Optional[str]:
        """Safe way of getting a metadata value, returns None if metadata or key does not exist"""
        return (self.metadata or {}).get(key)

    @property
    def type(self):
        """Shortcut for it's metadata-defined type (e.g. 'TransformerEnd')"""
        return self.metadata_value("type")

    @property
    def base_voltage(self) -> Optional[float]:
        """Gets the asset's base voltage as a float, or None if it has no base voltage"""
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

    def time_series(self, measurement_type: str = None, timeseries_type: str = None, **kwargs):
        """Retrieves the time series in the asset subtree.

        Args:
            measurement_type: Type of measurement, e.g. ThreePhaseActivePower
            timeseries_type: Type of time series, e.g. estimated_value
            kwargs: Other metadata filters
        """
        metadata_filter = {"measurement_type": measurement_type, "timeseries_type": timeseries_type, **kwargs}
        metadata_filter = {k: v for k, v in metadata_filter.items() if v}
        return self._cognite_client.time_series.list(asset_subtree_ids=[self.id], metadata=metadata_filter)

    @staticmethod
    def _load_from_asset(asset, class_name, cognite_client):
        cls = _str_to_class(class_name) or PowerAsset
        return cls(cognite_client=cognite_client, **asset.dump())

    @staticmethod
    def _sequence_number_filter(sequence_number) -> Optional[Callable]:
        if sequence_number is not None:
            if not isinstance(sequence_number, Iterable):
                sequence_number = [sequence_number]
            return lambda a: a.sequence_number in sequence_number


class Terminal(PowerAsset):
    def analogs(self) -> "PowerAssetList":
        """Shortcut for finding the associated Analogs for a Terminal"""
        return self.relationship_sources("Analog")

    @property
    def sequence_number(self) -> Optional[int]:
        """Gets the terminal's sequence number as an int.

        Returns:
            Optional[int]: The sequence number"""
        seq_number = self.metadata_value("Terminal.sequenceNumber")
        if seq_number is not None:
            return int(seq_number)
        else:
            return None

    def opposite_end(self) -> "Terminal":
        """Retrieves the opposite end (Sequence number 1 to 2 and vice versa), works on Terminals associated with both ACLineSegment and PowerTransformerEnd"""
        seq_number = self.sequence_number
        if seq_number not in [1, 2]:
            raise ValueError(f"Can't get opposite end for terminal with sequence number {seq_number}, should be 1 or 2")
        opposite_seq_number = 1 if seq_number == 2 else 2

        al = self.relationship_targets(power_type="ACLineSegment", relationship_type="connectsTo")
        if not al:
            al = self.relationship_targets(power_type="PowerTransformerEnd", relationship_type="connectsTo")
        if not al:
            raise SinglePowerAssetExpected(
                al, f"Could not find any ACLineSegment or PowerTransformerEnd connected to terminal {self.external_id}"
            )
        return assert_single_result(al[0].terminals(sequence_number=opposite_seq_number))


class Analog(PowerAsset):
    pass


class PowerTransformer(PowerAsset):
    def power_transformer_ends(
        self, end_number: Optional[Union[int, Iterable]] = None, base_voltage: Iterable = None
    ) -> "PowerAssetList":
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

    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the terminals on all the PowerTransformerEnds for a PowerTransformer"""
        return self.power_transformer_ends().terminals(sequence_number=sequence_number)

    def substation(self) -> "Substation":
        """Shortcut for finding the substation for a PowerTransformer"""
        return assert_single_result(self.relationship_targets("Substation"))


class Substation(PowerAsset):
    def power_transformers(self) -> "PowerAssetList":
        """Shortcut for finding the PowerTransformers for a Substation"""
        return self.relationship_sources("PowerTransformer")

    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the terminals for a substation"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", x_filter=filter)

    def ac_line_segments(self, base_voltage: Iterable = None) -> "PowerAssetList":
        """Shortcut for finding the connected ACLineSegments for a substation"""
        return self.terminals().ac_line_segments(base_voltage=base_voltage)

    def connected_substations(self, *args, **kwargs):
        """See PowerAssetList.connected_substations"""
        return PowerAssetList([self], cognite_client=self._cognite_client).connected_substations(*args, **kwargs)

    def distance(self, target: "Substation") -> int:
        """Shortest distance to target substation. Not efficient. Returns infinity if there is no connection."""
        distance = 0
        level_visit = {self}
        visited = {self}
        while level_visit:
            if target in level_visit:
                return distance
            level_pcl = PowerAssetList(list(level_visit), cognite_client=self._cognite_client)
            level_visit = set(level_pcl.connected_substations()) - visited
            visited.update(level_visit)
            distance += 1
        return np.inf


class PowerTransformerEnd(PowerAsset):
    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated Terminals"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=filter)

    def substation(self) -> Substation:
        """Shortcut for finding the substation for a PowerTransformerEnd"""
        return self.power_transformer().substation()

    def power_transformer(self) -> "PowerTransformer":
        """Shortcut for finding the PowerTransformer for a PowerTransformerEnd"""
        return assert_single_result(self.relationship_targets("PowerTransformer"))

    @property
    def end_number(self) -> Optional[int]:
        """Gets the PowerTransformerEnd's end number as an int."""
        end_number = self.metadata_value("TransformerEnd.endNumber")
        if end_number is not None:
            return int(end_number)
        else:
            return None

    def opposite_end(self):
        end_number = self.end_number
        if end_number not in [1, 2]:
            raise ValueError(
                f"Can't get opposite end for PowerTransformerEnd with end number {end_number}, should be 1 or 2"
            )
        opposite_end_number = 1 if end_number == 2 else 2
        return assert_single_result(self.power_transformer().power_transformer_ends(end_number=opposite_end_number))


class GeneratingUnit(PowerAsset):
    pass


class WindGeneratingUnit(GeneratingUnit):
    pass


class HydroGeneratingUnit(GeneratingUnit):
    pass


class SynchronousMachine(PowerAsset):
    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None):
        """Shortcut for finding the associated Terminals"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=filter)

    def generating_unit(self) -> GeneratingUnit:
        """Shortcut for finding the associated GeneratingUnit for a SynchronousMachine"""
        return PowerAsset._load_from_asset(
            assert_single_result([a for a in self.relationship_sources() if "GeneratingUnit" in a.type]),
            "GeneratingUnit",
            self._cognite_client,
        )

    def substation(self) -> Substation:
        """Shortcut for finding the associated substation for a SynchronousMachine"""
        return self.generating_unit().substation()


class ACLineSegment(PowerAsset):
    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated Terminals"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=filter)

    def substations(self) -> "PowerAssetList":
        """Shortcut for finding the connected substations"""
        return self.terminals().substations()


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    @property
    def type(self) -> Optional[str]:
        """Type of the list of assets, will raise a MixedPowerAssetListException if the list contains several asset types."""
        if not self.data:
            return None
        types = list({a.type for a in self.data})
        if len(types) != 1:
            raise MixedPowerAssetListException(
                f"Can not determine type of list with assets of types {', '.join(types)}"
            )
        return types[0]

    def has_type(self, check_type):
        return self.type == check_type

    def split_by_type(self) -> Dict[str, "PowerAssetList"]:
        """Returns a dictionary of asset type-> list of assets"""
        type_to_assets = defaultdict(lambda: PowerAssetList([], cognite_client=self._cognite_client))
        for asset in self.data:
            type_to_assets[asset.type].append(asset)
        return type_to_assets

    @staticmethod
    def _load_assets(
        assets, class_name, cognite_client, base_voltage: Iterable = None, x_filter: Callable = None
    ) -> "PowerAssetList":
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

    def power_transformer_ends(
        self, end_number: Optional[Union[int, Iterable]] = None, base_voltage: Optional[Iterable] = None
    ) -> "PowerAssetList":
        """Shortcut for finding the associated PowerTransformerEnds for a list of PowerTransformers or Terminals

        Args:
            end_number (Union[int, Iterable]): filter on transformer end number
            base_voltage (Iterable): filter on base voltage

        Returns:
            PowerAssetList: List of PowerTransformerEnd
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

    def power_transformers(self) -> "PowerAssetList":
        """Shortcut for finding the associated PowerTransformer for a list of PowerTransformerEnd or Substation"""
        if self.has_type("PowerTransformerEnd"):
            return self.relationship_targets("PowerTransformer")
        elif self.has_type("Substation"):
            return self.relationship_sources("PowerTransformer")
        elif not self.data:
            return PowerAssetList([])
        else:
            raise WrongPowerTypeError(f"Can't get PowerTransformers for a list of {self.type}")

    def substations(self) -> "PowerAssetList":
        """Shortcut for finding the associated Substations for a list of PowerTransformer, ACLineSegment or Terminal"""
        if self.has_type("PowerTransformer") or self.has_type("Terminal"):
            return self.relationship_targets("Substation")
        elif self.has_type("ACLineSegment"):
            return self.terminals().substations()
        elif not self.data:
            return PowerAssetList([])
        else:
            raise WrongPowerTypeError(f"Can't get substations for a list of {self.type}")

    def ac_line_segments(self, base_voltage: Iterable = None):
        """Shortcut for finding the associated ACLineSegment for a list of PowerTransformer, Substation or Terminal"""
        if self.has_type("PowerTransformer"):
            return self.substations().ac_line_segments(base_voltage=base_voltage)
        if self.has_type("Substation"):
            return self.terminals().ac_line_segments(base_voltage=base_voltage)
        elif self.has_type("Terminal"):
            return self.relationship_targets("ACLineSegment", relationship_type="connectsTo", base_voltage=base_voltage)
        elif not self.data:
            return PowerAssetList([])
        else:
            raise WrongPowerTypeError(f"Can't get ACLineSegments for a list of {self.type}")

    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None):
        """Shortcut for finding the associated Terminals. Works on lists with mixed asset types"""
        """Shortcut for finding the associated Terminals. For a power transformer list, will retrieve all terminals via terminal ends of the specified end_number(s)"""
        filter = PowerAsset._sequence_number_filter(sequence_number)
        try:
            if self.has_type("PowerTransformer"):
                return self.power_transformer_ends().terminals(sequence_number=sequence_number)
            elif self.has_type("Substation"):
                return self.relationship_sources("Terminal", x_filter=filter)
        except MixedPowerAssetListException:
            return PowerAssetList(
                sum([assets.terminals() for _type, assets in self.split_by_type().items()], []),
                cognite_client=self._cognite_client,
            )
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=filter)

    def analogs(self):
        """Shortcut for finding the associated Analogs. Only works when applied directly to a list of Terminals."""
        return self.relationship_sources("Analog")

    def opposite_ends(self):
        """Shortcut for finding the opposite ends for a list of PowerTransformerEnd or Terminals"""
        if self.has_type("PowerTransformerEnd"):
            end_numbers = {a.end_number for a in self.data}
            if end_numbers != {1} and end_numbers != {2}:
                raise ValueError(
                    f"Can't get opposite end for list with end number(s) {end_numbers}, should be either all 1 or all 2"
                )
            opposite_end_number = 1 if list(end_numbers)[0] == 2 else 2
            return self.power_transformers().power_transformer_ends(end_number=opposite_end_number)
        if self.has_type("Terminal"):
            seq_numbers = {a.sequence_number for a in self.data}
            if seq_numbers != {1} and seq_numbers != {2}:
                raise ValueError(
                    f"Can't get opposite end for list with sequence number(s) {seq_numbers}, should be either all 1 or all 2"
                )
            opposite_seq_number = 1 if list(seq_numbers)[0] == 2 else 2
            al = self.relationship_targets(power_type="ACLineSegment", relationship_type="connectsTo")
            if not al:
                al = self.relationship_targets(power_type="PowerTransformerEnd", relationship_type="connectsTo")
            return al.terminals(sequence_number=opposite_seq_number)
        else:
            raise WrongPowerTypeError(f"Can't get opposite ends for a list of {self.type}")

    def time_series(self, measurement_type=None, timeseries_type=None, **kwargs) -> TimeSeriesList:
        """Retrieves the time series in the asset subtrees.

        Args:
            measurement_type: Type of measurement, e.g. ThreePhaseActivePower
            timeseries_type: Type of time series, e.g. estimated_value
            kwargs: Other metadata filters"""
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

    def connected_substations(self, level=1, base_voltage=None):
        """Retrieves substations connected within level connections through ac_line_segments with base voltages within the specified range"""
        if not self.has_type("Substation"):
            raise ValueError(f"Can't get connected substations ends for a list of {self.type}")
        returned_substations = set(self.data)
        level_ss = self
        for i in range(level):
            level_ss = level_ss.ac_line_segments(base_voltage=base_voltage).substations()
            level_ss.data = [a for a in level_ss.data if a not in returned_substations]
            returned_substations.update(level_ss)
        return PowerAssetList(list(returned_substations), cognite_client=self._cognite_client)
