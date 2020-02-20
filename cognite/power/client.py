from cognite.client.experimental import CogniteClient
from cognite.power._api.generic_power import GenericPowerAPI


class PowerClient(CogniteClient):
    """Main entrypoint into Cognite Power SDK. All services are made available through this object.

    In addition to all functionality from the basic and experimental Cognite Python SDK, includes:
    * `.ac_line_segments`
    * `.substations`
    * `.transformers`
    * `.transformer_ends`
    * `.sync_machines`
    * `.terminals`
    Each of which has a `list` function which returns the specific assets only. See documentation for GenericPowerAPI for details.
    """

    def __init__(
        self,
        base_url="https://greenfield.cognitedata.com",
        project="powerdummy",
        client_name="Cognite Power SDK",
        *args,
        **kwargs,
    ):
        super().__init__(project=project, base_url=base_url, client_name=client_name, *args, **kwargs)
        self.ac_line_segments = GenericPowerAPI({"type": "ACLineSegment"}, self.config, self._API_VERSION, self)
        self.substations = GenericPowerAPI({"type": "Substation"}, self.config, self._API_VERSION, self)
        self.transformers = GenericPowerAPI({"type": "PowerTransformer"}, self.config, self._API_VERSION, self)
        self.transformer_ends = GenericPowerAPI({"type": "PowerTransformerEnd"}, self.config, self._API_VERSION, self)
        self.sync_machines = GenericPowerAPI({"type": "SynchronousMachines"}, self.config, self._API_VERSION, self)
        self.terminals = GenericPowerAPI({"type": "Terminal"}, self.config, self._API_VERSION, self)
