from typing import *

from cognite.client._api.assets import AssetsAPI
from cognite.power.data_classes import PowerAsset, PowerAssetList


class GenericPowerAPI(AssetsAPI):
    def __init__(self, metadata_filter, config, api_version, cognite_client):
        super().__init__(config, api_version, cognite_client)
        self.metadata_filter = metadata_filter

    def list(self, limit=None, **filters):
        filters["metadata"] = {**filters.get("metadata", {}), **self.metadata_filter}
        return PowerAssetList(
            [PowerAsset(cognite_client=self._cognite_client, **a.dump()) for a in super().list(limit=limit, **filters)],
            cognite_client=self._cognite_client,
        )
