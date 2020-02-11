from typing import *

from cognite.client.data_classes import Asset, AssetList, AssetUpdate


class PowerAsset(Asset):
    @property
    def type(self):
        return self.metadata.get("type")

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


class PowerAssetList(AssetList):
    _RESOURCE = PowerAsset
    _UPDATE = AssetUpdate

    def dump(self, camel_case: bool = False) -> List[Dict[str, Any]]:
        return [resource.dump(camel_case, expand=("type",)) for resource in self.data]
