import math
import sys
import warnings
from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd

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


def _remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class PowerAsset(Asset):
    """Extended asset class for Power related resources"""

    def metadata_value(self, key: str) -> Optional[str]:
        """Safe way of getting a metadata value, returns None if metadata or key does not exist"""
        return (self.metadata or {}).get(key)

    def float_metadata_value(self, key) -> Optional[float]:
        val = self.metadata_value(key)
        if val is not None:
            return float(val)
        else:
            return None

    @property
    def type(self):
        """Shortcut for it's metadata-defined type (e.g. 'TransformerEnd')"""
        return self.metadata_value("type")

    @property
    def base_voltage(self) -> Optional[float]:
        """Gets the asset's base voltage as a float, or None if it has no base voltage"""
        return self.float_metadata_value("BaseVoltage_nominalVoltage")

    @property
    def grid_type(self) -> Optional[str]:
        """Gets the asset's grid type, without 'GridTypeKind.' prefix."""
        gt = self.metadata_value("Equipment.gridType")
        return _remove_prefix(gt, "GridTypeKind.") if gt else None

    def relationship_sources(self, *args, **kwargs):
        """Shortcut for finding all assets that are a source, with the current asset as a target - see PowerAssetList.relationship_sources for arguments"""
        return PowerAssetList([self], cognite_client=self._cognite_client).relationship_sources(*args, **kwargs)

    def relationship_targets(self, *args, **kwargs):
        """Shortcut for finding all assets that are a target, with the current asset as a source - see PowerAssetList.relationship_targets for arguments"""
        return PowerAssetList([self], cognite_client=self._cognite_client).relationship_targets(*args, **kwargs)

    def analogs(self):
        """Shortcut for finding the associated Analogs (via it's terminals)"""
        return self.terminals().analogs()

    def time_series(
        self,
        measurement_type: Union[str, List[str]] = None,
        timeseries_type: Union[str, List[str]] = None,
        unit: str = None,
        **kwargs,
    ):
        """Retrieves the time series in the asset subtree.

        Args:
            measurement_type: Type of measurement, e.g. "ThreePhaseActivePower", or list thereof
            timeseries_type: Type of time series, e.g. "estimated_value", or list thereof
            unit: Unit of the time series, e.g. 'kV'.
            kwargs: Other metadata filters
        """
        return PowerAssetList([self], cognite_client=self._cognite_client).time_series(
            measurement_type=measurement_type, timeseries_type=timeseries_type, unit=unit, **kwargs
        )

    @staticmethod
    def _load_from_asset(asset, class_name, cognite_client):
        cls = _str_to_class(class_name) or PowerAsset
        power_asset = cls._load(asset.dump(), cognite_client=cognite_client)
        if cls is not PowerAsset:
            assert power_asset.type in [
                None,
                class_name,
            ], f"Tried to load an asset {power_asset.type} as a {cls.__name__}"
        return power_asset

    @staticmethod
    def _sequence_number_filter(sequence_number) -> Optional[Callable]:
        if sequence_number is not None:
            if not isinstance(sequence_number, Iterable):
                sequence_number = [sequence_number]
            return lambda a: a.sequence_number in sequence_number


class LoadDurationMixin:
    def load_duration_curve(
        self,
        start,
        end="now",
        terminal=1,
        measurement_type="ThreePhaseActivePower",
        timeseries_type="estimated_value",
        granularity="1h",
        dropna=True,
        index_granularity=0.1,
    ) -> "pd.DataFrame":
        """Calculates a load-duration curve.

        Args:
            start, end: string, timestamp or datetime for start and end, as in datapoints.retrieve
            terminal, measurement_type, timeseries_type: which measurement of which terminal to retrieve.
            granularity: granularity to be used in retrieving time series data.
            dropna: whether to drop NaN values / gaps
            index_granularity: spacing of the regularized return value in %, e.g. 0.1 gives back 1001 elements.

        Returns:
            pd.DataFrame: dataframe with a load duration curve
        """

        ts = assert_single_result(
            self.terminals(sequence_number=terminal).time_series(
                measurement_type=measurement_type, timeseries_type=timeseries_type
            )
        )
        df = self._cognite_client.datapoints.retrieve_dataframe(
            id=ts.id, start=start, end=end, granularity=granularity, aggregates=["interpolation"], complete="fill"
        )
        if dropna:
            df = df.dropna()
        index = np.linspace(start=0, stop=1, num=round(100 / index_granularity + 1))
        regular_spacing = np.interp(
            index, np.linspace(start=0, stop=1, num=df.shape[0]), df.iloc[:, 0].sort_values(ascending=False).values
        )
        return pd.DataFrame(index=index, data=regular_spacing, columns=[self.name])


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
            pte = self.relationship_targets(power_type="PowerTransformerEnd", relationship_type="connectsTo")
            if not pte:
                raise SinglePowerAssetExpected(
                    al,
                    f"Could not find any ACLineSegment or PowerTransformerEnd connected to terminal {self.external_id}",
                )
            return assert_single_result(
                assert_single_result(pte).opposite_end().terminals(sequence_number=opposite_seq_number)
            )
        return assert_single_result(assert_single_result(al).terminals(sequence_number=opposite_seq_number))


class Analog(PowerAsset):
    pass


class BusbarSection(PowerAsset):
    def substation(self) -> "Substation":
        """Shortcut for finding the substation for a PowerTransformer"""
        return assert_single_result(self.relationship_targets("Substation"))


class CurrentLimit(PowerAsset):
    @property
    def value(self) -> Optional[float]:
        return self.float_metadata_value("CurrentLimit.value")

    @property
    def limit_type(self) -> str:
        return self.name.split("@")[0]


class TemperatureCurve(PowerAsset):
    pass


class TemperatureCurveDependentLimit(PowerAsset):
    pass


class TemperatureCurveData(PowerAsset):
    @property
    def percent(self) -> Optional[float]:
        return self.float_metadata_value("TemperatureCurveData.percent")

    @property
    def temperature(self) -> Optional[float]:
        return self.float_metadata_value("TemperatureCurveData.temperature")


class OperationalLimitSet(PowerAsset):
    pass


class OperationalLimitType(PowerAsset):
    pass


class ShuntCompensator(PowerAsset):
    def substation(self) -> "Substation":
        """Shortcut for finding the substation for a ShuntCompensator"""
        return assert_single_result(self.relationship_targets("Substation"))


class StaticVarCompensator(PowerAsset):
    def substation(self) -> "Substation":
        """Shortcut for finding the substation for a StaticVarCompensator"""
        return assert_single_result(self.relationship_targets("Substation"))


class PetersenCoil(PowerAsset):
    def substation(self) -> "Substation":
        """Shortcut for finding the substation for a PetersenCoil"""
        return assert_single_result(self.relationship_targets("Substation"))


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
    def power_transformers(self, grid_type: Optional[str] = None) -> "PowerAssetList":
        """Shortcut for finding the PowerTransformers for a Substation"""
        return self.relationship_sources("PowerTransformer", grid_type=grid_type)

    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the terminals for a substation"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", x_filter=filter)

    def ac_line_segments(self, base_voltage: Iterable = None, grid_type: Optional[str] = None) -> "PowerAssetList":
        """Shortcut for finding the connected ACLineSegments for a substation"""
        return self.terminals().ac_line_segments(base_voltage=base_voltage, grid_type=grid_type)

    def busbar_sections(self, base_voltage: Iterable = None):
        return self.relationship_sources("BusbarSection", base_voltage=base_voltage)

    def conform_loads(self):
        return self.relationship_sources("ConformLoad")

    def non_conform_loads(self):
        return self.relationship_sources("NonConformLoad")

    def generating_units(self, power_type: Optional[Union[str, List[str]]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated GeneratingUnit for a Substation

        Args:
            power_type: type of generating unit, default is ["HydroGeneratingUnit","WindGeneratingUnit","ThermalGeneratingUnit"] """
        if power_type is None:
            power_type = ["HydroGeneratingUnit", "WindGeneratingUnit", "ThermalGeneratingUnit"]
        if isinstance(power_type, str):
            power_type = [power_type]
        return PowerAssetList(
            sum([self.relationship_sources(pt) for pt in power_type], []), cognite_client=self._cognite_client,
        )

    def hydro_generating_units(self) -> "PowerAssetList":
        """Shortcut for finding the associated HydroGeneratingUnits for a Substation"""
        return self.generating_units("HydroGeneratingUnit")

    def wind_generating_units(self) -> "PowerAssetList":
        """Shortcut for finding the associated WindGeneratingUnits for a Substation"""
        return self.generating_units("WindGeneratingUnit")

    def thermal_generating_units(self) -> "PowerAssetList":
        """Shortcut for finding the associated ThermalGeneratingUnits for a list of Substations"""
        return self.generating_units("ThermalGeneratingUnit")

    def connected_substations(self, *args, **kwargs) -> "PowerAssetList":
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


class PowerTransferCorridor(PowerAsset):
    def synchronous_machines(self) -> "PowerAssetList":
        """Shortcut for finding the associated SynchronousMachines for a PowerTransferCorridor.."""
        return self.relationship_sources("SynchronousMachine")

    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated Terminals"""
        filter = self._sequence_number_filter(sequence_number)
        return self.synchronous_machines().relationship_sources(
            "Terminal", relationship_type="connectsTo", x_filter=filter
        )


class PowerTransformerEnd(PowerAsset, LoadDurationMixin):
    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated Terminals"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=filter)

    @property
    def grid_type(self) -> Optional[str]:
        """Gets the asset's grid type, without 'GridTypeKind.' prefix."""
        gt = self.metadata_value("TransformerEnd.gridType")
        return _remove_prefix(gt, "GridTypeKind.") if gt else None

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
        """Gets the PowerTransformerEnd's opposite end (end number 1->2 and 2->1)"""
        end_number = self.end_number
        if end_number not in [1, 2]:
            raise ValueError(
                f"Can't get opposite end for PowerTransformerEnd with end number {end_number}, should be 1 or 2"
            )
        opposite_end_number = 1 if end_number == 2 else 2
        return assert_single_result(self.power_transformer().power_transformer_ends(end_number=opposite_end_number))


class GeneratingUnit(PowerAsset):
    def synchronous_machines(self) -> "PowerAssetList":
        """Shortcut for finding the associated SynchronousMachines for a list of generating units. NB: does not check types."""
        return self.relationship_sources("SynchronousMachine")


class WindGeneratingUnit(GeneratingUnit):
    pass


class HydroGeneratingUnit(GeneratingUnit):
    pass


class ThermalGeneratingUnit(GeneratingUnit):
    pass


class ConformLoad(PowerAsset):
    pass


class NonConformLoad(PowerAsset):
    pass


class SynchronousMachine(PowerAsset, LoadDurationMixin):
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


class ACLineSegment(PowerAsset, LoadDurationMixin):
    def terminals(self, sequence_number: Optional[Union[int, Iterable]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated Terminals"""
        filter = self._sequence_number_filter(sequence_number)
        return self.relationship_sources("Terminal", relationship_type="connectsTo", x_filter=filter)

    def substations(self) -> "PowerAssetList":
        """Shortcut for finding the connected substations"""
        return self.terminals().substations()

    def operational_limit_sets(self) -> "PowerAssetList":
        return self.relationship_sources("OperationalLimitSet")

    def current_limits(self) -> "PowerAssetList":
        return self.operational_limit_sets().relationship_sources("CurrentLimit")

    def temperature_curve_dependent_limits(self) -> "PowerAssetList":
        return self.relationship_sources("TemperatureCurveDependentLimit")

    def temperature_curves(self) -> "PowerAssetList":
        return self.temperature_curve_dependent_limits().relationship_sources("TemperatureCurve")

    def temperature_curve_data(self) -> "PowerAssetList":
        return self.temperature_curves().relationship_sources("TemperatureCurveData")

    def current_limits_overview(self) -> "pd.DataFrame":
        """Returns a dataframe which combines data from `temperature_curve_data` and `current_limits` """
        current_limits = self.current_limits()
        temp_coefficients = self.temperature_curve_data()
        return pd.DataFrame(
            {
                "type": current_limit.limit_type,
                "temperature": coeff.temperature,
                "limit": current_limit.value * coeff.percent,
                "name": self.name,
            }
            for current_limit in current_limits
            for coeff in temp_coefficients
        ).set_index(["name", "type", "temperature"])


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    @property
    def type(self) -> Optional[str]:
        """Type of the list of assets, will raise a MixedPowerAssetListException if the list contains several asset types."""
        if not self.data:
            return None

        types = list({a.type for a in self.data})
        if None in types:
            raise MixedPowerAssetListException("One or more assets do not have a valid power type")
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

    def filter(
        self, base_voltage: Iterable = None, grid_type: str = None, x_filter: Callable = None,
    ) -> "PowerAssetList":
        """Filters list by grid type, base_voltage or arbitrary function of asset->bool"""
        power_assets = self.data
        if x_filter:
            power_assets = [a for a in power_assets if x_filter(a)]
        if base_voltage is not None:
            power_assets = [a for a in power_assets if a.base_voltage in base_voltage]
        if grid_type is not None:
            power_assets = [a for a in power_assets if a.grid_type == grid_type]
        return PowerAssetList(power_assets, cognite_client=self._cognite_client)

    def to_power_area(
        self, interior_substation: Union[str, Substation] = None, grid_type: str = None, base_voltage: Iterable = None,
    ):
        try:
            if self.type == "Substation":
                return self._cognite_client.power_area(self)
            elif self.type == "ACLineSegment":
                if interior_substation:
                    return self._cognite_client.power_area(
                        ac_line_segments=self,
                        interior_substation=interior_substation,
                        grid_type=grid_type,
                        base_voltage=base_voltage,
                    )
                else:
                    raise ValueError("Need an substation on the interior of the area to create from ac line segments.")
            else:
                raise ValueError("PowerArea can only be created from a list of Substations, not {}s.".format(self.type))
        except MixedPowerAssetListException as e:
            raise ValueError("PowerArea can only be created from a list of Substations, not mixed power assets.") from e

    @staticmethod
    def _load_assets(
        assets,
        class_name,
        cognite_client,
        base_voltage: Iterable = None,
        grid_type: str = None,
        x_filter: Callable = None,
    ) -> "PowerAssetList":
        power_assets = [PowerAsset._load_from_asset(a, class_name, cognite_client) for a in assets]
        return PowerAssetList(power_assets, cognite_client=cognite_client).filter(
            base_voltage=base_voltage, grid_type=grid_type, x_filter=x_filter
        )

    @staticmethod
    def _filter_and_convert(
        client,
        external_ids,
        power_type,
        base_voltage: Iterable = None,
        grid_type: str = None,
        x_filter: Callable = None,
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
            assets, power_type, cognite_client=client, base_voltage=base_voltage, grid_type=grid_type, x_filter=x_filter
        )

    def relationships(
        self,
        power_type: str = None,
        relationship_type: str = "belongsTo",
        base_voltage: Optional[Iterable] = None,
        grid_type: Optional[str] = None,
        x_filter: Optional[Callable] = None,
        _sources=None,
        _targets=None,
    ) -> "PowerAssetList":
        """Internal function responsible for finding assets connected by relationships. 

        Args:
            power_type (str): Type of asset, e.g. PowerTransformer
            relationship_type (str): Type of relationship, typically belongsTo or connectsTo
            base_voltage (str):
            grid_type (str):
            x_filter (Callable): Other filter to be applied to the asset, returns those for which x_filter(asset) is truthy after converting the asset to the proper type
            _sources, _targets: internally passed by relationship_sources/relationship_targets
        Returns:
            PowerAssetList: list of connected assets
        """
        if _sources and _targets:
            raise ValueError("Can not combine _sources and _targets.")
        if not _sources and not _targets:
            return PowerAssetList([], cognite_client=self._cognite_client)
        rels = self._cognite_client.relationships_playground.list(
            sources=_sources, targets=_targets, relationship_type=relationship_type, limit=None
        )
        if _sources:
            asset_ids = [r.target["resourceId"] for r in rels]
        else:
            asset_ids = [r.source["resourceId"] for r in rels]

        return PowerAssetList._filter_and_convert(
            self._cognite_client,
            asset_ids,
            power_type,
            base_voltage=base_voltage,
            grid_type=grid_type,
            x_filter=x_filter,
        )

    def relationship_sources(self, *args, **kwargs) -> "PowerAssetList":
        """Shortcut for finding all assets that are a source, with the current assets as targets. See PowerAssetList.relationships for list of arguments."""
        return self.relationships(
            _sources=None, _targets=[{"resource": "Asset", "resourceId": a.external_id} for a in self], *args, **kwargs
        )

    def relationship_targets(self, *args, **kwargs) -> "PowerAssetList":
        """Shortcut for finding all assets that are a target, with the current assets as sources. See PowerAssetList.relationships for list of arguments."""
        return self.relationships(
            _sources=[{"resource": "Asset", "resourceId": a.external_id} for a in self], _targets=None, *args, **kwargs
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

    def power_transformers(self, grid_type: Optional[str] = None) -> "PowerAssetList":
        """Shortcut for finding the associated PowerTransformer for a list of PowerTransformerEnd or Substation"""
        if self.has_type("PowerTransformerEnd"):
            return self.relationship_targets("PowerTransformer", grid_type=grid_type)
        elif self.has_type("Substation"):
            return self.relationship_sources("PowerTransformer", grid_type=grid_type)
        elif not self.data:
            return PowerAssetList([])
        else:
            raise WrongPowerTypeError(f"Can't get PowerTransformers for a list of {self.type}")

    def generating_units(self, power_type: Optional[Union[str, List[str]]] = None) -> "PowerAssetList":
        """Shortcut for finding the associated GeneratingUnit for a list of Substations

        Args:
            power_type: type of generating unit, default is ["HydroGeneratingUnit","WindGeneratingUnit","ThermalGeneratingUnit"] """
        if power_type is None:
            power_type = ["HydroGeneratingUnit", "WindGeneratingUnit", "ThermalGeneratingUnit"]
        if isinstance(power_type, str):
            power_type = [power_type]
        if self.has_type("Substation"):
            return PowerAssetList(
                sum([self.relationship_sources(pt) for pt in power_type], []), cognite_client=self._cognite_client,
            )
        elif not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        else:
            raise WrongPowerTypeError(f"Can't get Generating Units [{power_type}] for a list of {self.type}")

    def synchronous_machines(self) -> "PowerAssetList":
        """Shortcut for finding the associated SynchronousMachines for a list of generating units. NB: does not check types."""
        return self.relationship_sources("SynchronousMachine")

    def hydro_generating_units(self) -> "PowerAssetList":
        """Shortcut for finding the associated HydroGeneratingUnits for a list of Substations"""
        return self.generating_units("HydroGeneratingUnit")

    def wind_generating_units(self) -> "PowerAssetList":
        """Shortcut for finding the associated WindGeneratingUnits for a list of Substations"""
        return self.generating_units("WindGeneratingUnit")

    def thermal_generating_units(self) -> "PowerAssetList":
        """Shortcut for finding the associated ThermalGeneratingUnits for a list of Substations"""
        return self.generating_units("ThermalGeneratingUnit")

    def busbar_sections(self, base_voltage: Iterable = None):
        if self.has_type("Substation"):
            return self.relationship_sources("BusbarSection", base_voltage=base_voltage)
        elif not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        else:
            raise WrongPowerTypeError(f"Can't get ConformLoads for a list of {self.type}")

    def conform_loads(self, base_voltage: Iterable = None) -> "PowerAssetList":
        if self.has_type("Substation"):
            return self.relationship_sources("ConformLoad", base_voltage=base_voltage)
        elif not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        else:
            raise WrongPowerTypeError(f"Can't get ConformLoads for a list of {self.type}")

    def non_conform_loads(self, base_voltage: Iterable = None) -> "PowerAssetList":
        if self.has_type("Substation"):
            return self.relationship_sources("NonConformLoad", base_voltage=base_voltage)
        elif not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        else:
            raise WrongPowerTypeError(f"Can't get NonConformLoad for a list of {self.type}")

    def substations(self) -> "PowerAssetList":
        """Shortcut for finding the associated Substations for a list of PowerTransformer, GeneratingUnit, (Non)ConformLoad, ACLineSegment, BusbarSection Shunt/StaticVarCompensator, PetersenCoil or Terminal"""
        if (
            self.has_type("PowerTransformer")
            or self.has_type("Terminal")
            or self.has_type("ConformLoad")
            or self.has_type("NonConformLoad")
            or self.has_type("WindGeneratingUnit")
            or self.has_type("ThermalGeneratingUnit")
            or self.has_type("HydroGeneratingUnit")
            or self.has_type("BusbarSection")
            or self.has_type("ShuntCompensator")
            or self.has_type("StaticVarCompensator")
            or self.has_type("PetersenCoil")
        ):
            return self.relationship_targets("Substation")
        elif self.has_type("ACLineSegment"):
            return self.terminals().substations()
        elif not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
        else:
            raise WrongPowerTypeError(f"Can't get substations for a list of {self.type}")

    def ac_line_segments(self, base_voltage: Iterable = None, grid_type: Optional[str] = None):
        """Shortcut for finding the associated ACLineSegment for a list of PowerTransformer, Substation or Terminal"""
        if self.has_type("Substation"):
            return self.terminals().ac_line_segments(base_voltage=base_voltage, grid_type=grid_type)
        elif self.has_type("Terminal"):
            return self.relationship_targets(
                "ACLineSegment", relationship_type="connectsTo", base_voltage=base_voltage, grid_type=grid_type
            )
        elif not self.data:
            return PowerAssetList([], cognite_client=self._cognite_client)
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
                pte = self.relationship_targets(power_type="PowerTransformerEnd", relationship_type="connectsTo")
                return pte.opposite_ends().terminals(sequence_number=opposite_seq_number)
            return al.terminals(sequence_number=opposite_seq_number)
        else:
            raise WrongPowerTypeError(f"Can't get opposite ends for a list of {self.type}")

    def time_series(
        self,
        measurement_type: Union[str, List[str]] = None,
        timeseries_type: Union[str, List[str]] = None,
        unit: str = None,
        **kwargs,
    ) -> TimeSeriesList:
        """Retrieves the time series in the asset subtrees.

        Args:
            measurement_type: Type of measurement, e.g. "ThreePhaseActivePower", or list thereof
            timeseries_type: Type of time series, e.g. "estimated_value", or list thereof
            unit: Unit of the time series, e.g. 'kV'.
            kwargs: Other metadata filters"""
        if isinstance(measurement_type, str):
            measurement_type = [measurement_type]
        if isinstance(timeseries_type, str):
            timeseries_type = [timeseries_type]
        tasks = []
        for ts_type in timeseries_type or [None]:
            for mt_type in measurement_type or [None]:
                metadata_filter = {"measurement_type": mt_type, "timeseries_type": ts_type, **kwargs}
                metadata_filter = {k: v for k, v in metadata_filter.items() if v}
                chunk_size = 100
                ids = [a.id for a in self.data]
                tasks.extend(
                    [
                        {
                            "asset_subtree_ids": ids[i : i + self._retrieve_chunk_size],
                            "unit": unit,
                            "metadata": metadata_filter,
                            "limit": None,
                        }
                        for i in range(0, len(ids), chunk_size)
                    ]
                )
        res_list = execute_tasks_concurrently(self._cognite_client.time_series.list, tasks, max_workers=10)
        return TimeSeriesList(sum(res_list.joined_results(), []))

    def connected_substations(
        self,
        level: int = 1,
        exact: bool = False,
        include_lines=False,
        base_voltage: Iterable = None,
        grid_type: Optional[str] = None,
    ) -> "PowerAssetList":
        """Retrieves substations connected within level connections through ac_line_segments with base voltages within the specified range

        Args:
                level: number of connections to traverse
                exact: only return substations whose minimum distance is exactly level
                include_lines: also return ACLineSegments that make up the connections. Can not be used in combination with exact.
                base_voltage: only consider ACLineSegments with these base voltage
                grid_type:  only consider ACLineSegments of this grid type
        """
        if exact and include_lines:
            raise ValueError("Can not include lines when an exact distance is requested")
        if not self.has_type("Substation"):
            raise WrongPowerTypeError(f"Can't get connected substations for a list of {self.type}")
        visited_substations = set(self.data)
        visited_lines = set()
        substations_at_level = self
        for i in range(level):
            ac_line_segments = substations_at_level.ac_line_segments(base_voltage=base_voltage, grid_type=grid_type)
            substations_at_level = ac_line_segments.substations()
            substations_at_level.data = [a for a in substations_at_level.data if a not in visited_substations]
            if not substations_at_level:
                break
            visited_lines.update(ac_line_segments)
            visited_substations.update(substations_at_level)
        if exact:
            returned_assets = substations_at_level
        elif include_lines:
            returned_assets = list(visited_substations) + list(visited_lines)
        else:
            returned_assets = visited_substations
        return PowerAssetList(list(returned_assets), cognite_client=self._cognite_client)

    def current_limits_overview(self) -> "pd.DataFrame":
        """See ACLineSegment#current_limits_overview"""
        if not self.has_type("ACLineSegment"):
            raise WrongPowerTypeError(f"Can't get connected current limits dataframe for a list of {self.type}")
        res_list = execute_tasks_concurrently(
            ACLineSegment.current_limits_overview, [(a,) for a in self], max_workers=10
        )
        return pd.concat(res_list.joined_results())

    def load_duration_curve(
        self,
        start,
        end="now",
        terminal=1,
        measurement_type="ThreePhaseActivePower",
        timeseries_type="estimated_value",
        granularity="1h",
        dropna=True,
        index_granularity=0.1,
    ) -> "pd.DataFrame":
        """See ACLineSegment#load_duration_curve"""
        if self.type not in ["ACLineSegment", "PowerTransformerEnd", "SynchronousMachine"]:  # , "PowerTransferCorridor"
            raise WrongPowerTypeError(f"Can't get load duration curves dataframe for a list of {self.type}")
        if len(self.data) > 1000:
            raise ValueError("Too many line segments in this list to get load duration curves")
        res_list = execute_tasks_concurrently(
            ACLineSegment.load_duration_curve,
            [
                (a, start, end, terminal, measurement_type, timeseries_type, granularity, dropna, index_granularity)
                for a in self
            ],
            max_workers=10,
        )
        return pd.concat(res_list.joined_results(), axis=1)
