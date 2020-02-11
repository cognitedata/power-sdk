from cognite.client.experimental import CogniteClient
from cognite.power._api.generic_power import GenericPowerAPI


class PowerClient(CogniteClient):
    def __init__(
        self,
        base_url="https://greenfield.cognitedata.com",
        project="powerdummy",
        client_name="Cognite Power SDK",
        *args,
        **kwargs,
    ):
        super().__init__(base_url=base_url, client_name=client_name, *args, **kwargs)
        self.ac_line_segments = GenericPowerAPI({"type": "ACLineSegment"}, self.config, self._API_VERSION, self)
        self.substations = GenericPowerAPI({"type": "Substation"}, self.config, self._API_VERSION, self)
        self.transformers = GenericPowerAPI({"type": "PowerTransformer"}, self.config, self._API_VERSION, self)
