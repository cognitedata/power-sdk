from cognite.power._api.generic_power import GenericPowerAPI
from cognite.power.data_classes import PowerAsset, PowerAssetList


class PowerTransformerEndsAPI(GenericPowerAPI):
    def __init__(self, config, api_version, cognite_client):
        super().__init__("PowerTransformerEnd", config, api_version, cognite_client, "TransformerEnd.gridType")

    def list(self, end_number=None, *args, **kwargs) -> PowerAssetList:
        if end_number:
            kwargs["metadata"] = kwargs.get("metadata", {})
            kwargs["metadata"]["TransformerEnd.endNumber"] = str(end_number)
        return super().list(*args, **kwargs)

    def search(self, end_number=None, *args, **kwargs) -> PowerAssetList:
        if end_number:
            kwargs["metadata"] = kwargs["metadata"] or {}
            kwargs["metadata"]["TransformerEnd.endNumber"] = str(end_number)
        return super().search(*args, **kwargs)
