from cognite.client.experimental import CogniteClient
from cognite.power._api.generic_power import GenericPowerAPI


class PowerClient(CogniteClient):
    """Main entrypoint into Cognite Power SDK. All services are made available through this object.

    In addition to all functionality from the basic and experimental Cognite Python SDK, includes:
    * `.ac_line_segments`
    * `.substations`
    * `.synchronous_machines`
    * `.hydro_generating_units`, `.wind_generating_units`
    * `.power_transformers`
    * `.power_transformer_ends`
    * `.terminals`
    * `.analogs`
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

        self.ac_line_segments = GenericPowerAPI("ACLineSegment", self.config, self._API_VERSION, self)
        self.substations = GenericPowerAPI("Substation", self.config, self._API_VERSION, self)
        self.synchronous_machines = GenericPowerAPI("SynchronousMachine", self.config, self._API_VERSION, self)

        self.hydro_generating_units = GenericPowerAPI("HydroGeneratingUnit", self.config, self._API_VERSION, self)
        self.wind_generating_units = GenericPowerAPI("WindGeneratingUnit", self.config, self._API_VERSION, self)

        self.power_transformers = GenericPowerAPI("PowerTransformer", self.config, self._API_VERSION, self)
        self.power_transformer_ends = GenericPowerAPI(
            "PowerTransformerEnd", self.config, self._API_VERSION, self, grid_type_field="TransformerEnd.gridType"
        )

        self.terminals = GenericPowerAPI("Terminal", self.config, self._API_VERSION, self)
        self.analogs = GenericPowerAPI("Analog", self.config, self._API_VERSION, self)


#        self.shunt_compensators = GenericPowerAPI("ShuntCompensator", self.config, self._API_VERSION, self)
#        self.static_var_compensators = GenericPowerAPI("StaticVarCompensator", self.config, self._API_VERSION, self)
#        self.peterson_coils = GenericPowerAPI("PetersenCoil", self.config, self._API_VERSION, self)
