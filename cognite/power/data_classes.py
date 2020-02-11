import warnings
from typing import *

from cognite.client.data_classes import Asset, AssetList, AssetUpdate
from cognite.power.exceptions import WrongPowerTypeError


class PowerAsset(Asset):
    @property
    def type(self):
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
        rels = self._cognite_client.relationships.list(
            source_resource=resource_type,
            target_resource="Asset",
            target_resource_id=self.external_id,
            relationship_type=relationship_type,
            limit=None,
        )
        return self._filter_and_convert([r.source["resourceId"] for r in rels], power_type)

    def relationship_targets(self, power_type, resource_type="Asset", relationship_type="belongsTo"):
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

    def transformer_end(self):
        if self.type not in ["PowerTransformer"]:
            return self.relationship_targets("PowerTransformer")[0]

    def substation(self):
        if self.type == "PowerTransformerEnd":
            return self.relationship_sources("PowerTransformer")[0].substation()
        if self.type not in ["PowerTransformer"]:
            raise WrongPowerTypeError("A PowerAsset of type {} does not have a substation".format(self.type))
        return self.relationship_sources("Substation")[0]


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    def dump(self, camel_case: bool = False) -> List[Dict[str, Any]]:
        return [resource.dump(camel_case, expand=("type",)) for resource in self.data]

    @staticmethod
    def _load_assets(assets, cognite_client):
        return PowerAssetList(
            [PowerAsset(cognite_client=cognite_client, **a.dump()) for a in assets], cognite_client=cognite_client
        )
