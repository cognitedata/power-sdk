from typing import *

from cognite.experimental import CogniteClient
from cognite.power._api.generic_power import GenericPowerAPI
from cognite.power._api.power_transformer_ends_api import PowerTransformerEndsAPI
from cognite.power.data_classes import ACLineSegment, Substation
from cognite.power.power_area import PowerArea
from cognite.power.power_graph import PowerGraph


class PowerClient(CogniteClient):
    """Main entrypoint into Cognite Power SDK. All services are made available through this object.

    In addition to all functionality from the basic and experimental Cognite Python SDK, includes APIs for:

        * ``.ac_line_segments``
        * ``.substations``
        * ``.synchronous_machines``
        * ``.hydro_generating_units``
        * ``.wind_generating_units``
        * ``.thermal_generating_units``
        * ``.power_transformers``
        * ``.power_transformer_ends``
        * ``.conform_loads``
        * ``.terminals``
        * ``.analogs``
        * ``.power_assets``: does not filter by type.

    Each of which is a GenericPowerAPI which returns assets of the relevant type(s).
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
        self.thermal_generating_units = GenericPowerAPI("ThermalGeneratingUnit", self.config, self._API_VERSION, self)
        self.conform_loads = GenericPowerAPI("ConformLoad", self.config, self._API_VERSION, self)

        self.power_transformers = GenericPowerAPI("PowerTransformer", self.config, self._API_VERSION, self)
        self.power_transformer_ends = PowerTransformerEndsAPI(self.config, self._API_VERSION, self)

        self.terminals = GenericPowerAPI("Terminal", self.config, self._API_VERSION, self)
        self.analogs = GenericPowerAPI("Analog", self.config, self._API_VERSION, self)

        self.power_assets = GenericPowerAPI(None, self.config, self._API_VERSION, self)
        self.power_graph = None

    #        self.shunt_compensators = GenericPowerAPI("ShuntCompensator", self.config, self._API_VERSION, self)
    #        self.static_var_compensators = GenericPowerAPI("StaticVarCompensator", self.config, self._API_VERSION, self)
    #        self.peterson_coils = GenericPowerAPI("PetersenCoil", self.config, self._API_VERSION, self)

    def initialize_power_graph(self):
        if not self.power_graph:
            self.power_graph = PowerGraph(self)

    def power_area(
        self,
        substations: List[Union[Substation, str]] = None,
        ac_line_segments: List[Union[ACLineSegment, str, Tuple[str, str]]] = None,
        interior_substation: Substation = None,
        grid_type: str = None,
        base_voltage: Iterable = None,
    ):
        self.initialize_power_graph()
        if substations:
            return PowerArea(self, substations, self.power_graph)
        elif ac_line_segments and interior_substation:
            return PowerArea.from_interface(
                self, self.power_graph, ac_line_segments, interior_substation, grid_type, base_voltage
            )
        else:
            raise ValueError(
                "Need either a list of substations or a list of ac_line_segments plus an interior substation to define an area"
            )
