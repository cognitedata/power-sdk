import warnings
from typing import *

from cognite.client.data_classes import Asset, AssetList, AssetUpdate
from cognite.power.exceptions import WrongPowerTypeError, assert_single_result


class PowerAsset(Asset):
    """Extended asset class for Power related resources"""

    @property
    def type(self):
        """Shortcut for it's metadata-defined type (e.g. 'TransformerEnd')"""
        return (self.metadata or {}).get("type")

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

    def relationship_sources(self, power_type, resource_type="Asset", relationship_type="belongsTo"):
        """Shortcut for finding all assets that are a source, with the current asset as a target"""
        rels = self._cognite_client.relationships.list(
            source_resource=resource_type,
            target_resource="Asset",
            target_resource_id=self.external_id,
            relationship_type=relationship_type,
            limit=None,
        )
        return self._filter_and_convert([r.source["resourceId"] for r in rels], power_type)

    def relationship_targets(self, power_type, resource_type="Asset", relationship_type="belongsTo"):
        """Shortcut for finding all assets that are a target, with the current asset as a source"""
        rels = self._cognite_client.relationships.list(
            target_resource=resource_type,
            source_resource="Asset",
            source_resource_id=self.external_id,
            relationship_type=relationship_type,
            limit=None,
        )
        return self._filter_and_convert([r.target["resourceId"] for r in rels], power_type)

    def _filter_and_convert(self, external_ids, power_type):
        assets = self._cognite_client.assets.retrieve_multiple(external_ids=external_ids, ignore_unknown_ids=True)
        if len(assets) != len(external_ids):
            warnings.warn(
                "{} assets not found when looking up {}s among {}".format(
                    len(external_ids) - len(assets), power_type, external_ids
                )
            )
        if power_type:
            assets = [a for a in assets if (a.metadata or {}).get("type") == power_type]
        return PowerAssetList._load_assets(assets, cognite_client=self._cognite_client)

    def terminal(self):
        """Shortcut for finding the associated Terminal for a Substation, PowerTransformerEnd, or SynchronousMachine"""
        if self.type == "PowerTransformer":
            return self.transformer_end().terminal()
        if self.type not in ["Substation", "PowerTransformerEnd", "SynchronousMachine"]:
            raise WrongPowerTypeError("A PowerAsset of type {} does not have a Terminal".format(self.type))
        return assert_single_result(self.relationship_sources("Terminal"))

    def analogs(self):
        """Shortcut for finding the associated Analogs for a Terminal (or any PowerAsset which has a Terminal)"""
        if self.type != "Terminal":
            return self.terminal().analogs()
        return self.relationship_sources("Analog")

    def transformer_end(self):
        """Shortcut for finding the associated PowerTransformerEnd for a PowerTransformer"""
        if self.type not in ["PowerTransformer"]:
            raise WrongPowerTypeError("A PowerAsset of type {} does not have a PowerTransformerEnd".format(self.type))
        return assert_single_result(self.relationship_targets("PowerTransformerEnd"))

    def generator(self):
        if self.type != "SynchronousMachine":
            raise WrongPowerTypeError(
                "Can only find the power generator for a SynchronousMachine, not for a  {}.".format(self.type)
            )
        return assert_single_result([a for a in self.relationship_sources() if "Generator" in a.type])

    def substation(self):
        """Shortcut for finding the associated transformer for a PowerTransformer, PowerTransformerEnd, .."""
        if self.type == "PowerTransformerEnd":
            return assert_single_result(self.relationship_targets("PowerTransformer")).substation()
        if self.type == "SynchronousMachine":
            return self.generator().substation()
        # if self.type != ["PowerTransformer"] and not slef.:
        #    raise WrongPowerTypeError("A PowerAsset of type {} does not have a substation".format(self.type))
        return assert_single_result(self.relationship_targets("Substation"))

    def line_segments(self):
        """Shortcut for finding the connected ACLineSegments for a substation (or associated terminal)"""
        if self.type == "Substation":
            return self.terminal().line_segments()
        if self.type != "Terminal":
            raise WrongPowerTypeError("Can only find the lines for a substation, not for a  {}.".format(self.type))
        return self.relationship_targets("ACLineSegment", relationship_type="connectsTo")

    def connected_terminals(self):
        """Shortcut for finding the connected Terminals for an ACLineSegment"""
        if self.type != "ACLineSegment":
            raise WrongPowerTypeError(
                "Can only find connected terminals to an ACLineSegment, not a {}.".format(self.type)
            )
        return self.relationship_sources("Terminal", relationship_type="connectsTo")

    def time_series(self):
        if self.type != "Terminal":
            return self.terminal().time_series()
        return super().time_series()

    @staticmethod
    def _load_from_asset(asset, cognite_client):
        return PowerAsset(cognite_client=cognite_client, **asset.dump())


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    def dump(self, camel_case: bool = False) -> List[Dict[str, Any]]:
        return [resource.dump(camel_case, expand=("type",)) for resource in self.data]

    @staticmethod
    def _load_assets(assets, cognite_client):
        return PowerAssetList(
            [PowerAsset._load_from_asset(a, cognite_client) for a in assets], cognite_client=cognite_client
        )
