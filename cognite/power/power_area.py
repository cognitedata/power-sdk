import networkx as nx

from cognite.power.data_classes import *
from cognite.power.power_plot import PowerPlot


class PowerArea:
    """
    Describes the electrical grid in a connected set of substations.
    """

    def __init__(self, cognite_client, substations: List[Union[Substation, str]], power_graph):
        substations = [n.name if isinstance(n, Asset) else n for n in substations]
        self._cognite_client = cognite_client
        if len(substations) < 1:
            raise ValueError("A power area must have at least one substation")
        self._power_graph = power_graph
        self._graph = nx.MultiGraph()
        self._graph.add_nodes_from(power_graph.helpful_substation_lookup(k) for k in substations)
        self._graph.add_edges_from(
            (
                edge
                for edge in power_graph.graph.edges(data=True)
                if self._graph.has_node(edge[0]) and self._graph.has_node(edge[1])
            )
        )

    @classmethod
    def from_interface(
        cls,
        cognite_client,
        power_graph,
        ac_line_segments: List[Union[ACLineSegment, str, Tuple[str, str]]],
        interior_station: Union[Substation, str],
        grid_type: str = None,
        base_voltage: Iterable = None,
    ):
        """Creates a power area from a list of ac line segments, interpreted as the interface of the area, as well as an interior substation"""
        interior_station = interior_station.name if isinstance(interior_station, Substation) else interior_station
        ac_line_segments = [acls.name if isinstance(acls, Asset) else acls for acls in ac_line_segments]
        acls_edge_map = {edge[2]["object"].name: (edge[0], edge[1]) for edge in power_graph.graph.edges.data()}
        interface_edges = [acls_edge_map[acls] if isinstance(acls, str) else acls for acls in ac_line_segments]
        # create a copy of the networkx graph, remove the interface edges, use the copy to expand the area:
        full_networkx_graph = power_graph.graph
        temp_graph = full_networkx_graph.copy()
        for edge in interface_edges:
            while temp_graph.has_edge(*edge):
                temp_graph.remove_edge(*edge)
        power_graph.graph = temp_graph
        # some extra safety to ensure we always re-instate the original graph:
        try:
            power_area = cls(cognite_client, [interior_station], power_graph).expand_area(
                level=len(temp_graph), grid_type=grid_type, base_voltage=base_voltage
            )
        finally:
            # re-instate the original networkx graph:
            power_graph.graph = full_networkx_graph
        for edge in interface_edges:
            if edge[0] in power_area._graph.nodes and edge[1] in power_area._graph.nodes:
                raise ValueError("Inconsistent interface. The interface must create an isolated zone.")
        return power_area

    @staticmethod
    def _graph_substations(graph, client):
        return PowerAssetList([obj for n, obj in graph.nodes(data="object")], cognite_client=client)

    def substations(self) -> PowerAssetList:
        """Returns the list of Substations in the graph"""
        return self._graph_substations(self._graph, self._cognite_client)

    @staticmethod
    def _graph_ac_line_segments(graph, client):
        return PowerAssetList([obj for f, t, obj in graph.edges(data="object")], cognite_client=client)

    def ac_line_segments(self) -> PowerAssetList:
        """Returns the list of ACLineSegments in the graph"""
        return self._graph_ac_line_segments(self._graph, self._cognite_client)

    def _interface_edges(self, base_voltage: Iterable = None) -> List:
        return [  # select edges with either from or to in graph but not both
            (f, t, data)
            for f, t, data in self._power_graph.graph.edges(data=True)
            if (f in self._graph) ^ (t in self._graph)
            and (base_voltage is None or data["object"].base_voltage in base_voltage)
        ]

    def interface(self, base_voltage: Iterable = None) -> PowerAssetList:
        """Return the list of ACLineSegments going in/out of the area."""
        ac_line_segments = [data["object"] for f, t, data in self._interface_edges(base_voltage)]
        return PowerAssetList(ac_line_segments, cognite_client=self._cognite_client)

    def interface_terminals(self, base_voltage: Iterable = None) -> Tuple[PowerAssetList, PowerAssetList]:
        """Return the lists of Terminals on the inside and outside of the interface going in/out of the area."""
        inside_outside_terminals = [
            (data["terminals"][f], data["terminals"][t])
            if f in self._graph
            else (data["terminals"][t], data["terminals"][f])
            for f, t, data in self._interface_edges(base_voltage)
        ]
        return tuple(
            PowerAssetList([io[i] for io in inside_outside_terminals], cognite_client=self._cognite_client)
            for i in [0, 1]
        )

    def _find_neighbors(self, substation, grid_type, base_voltage):
        """finds the neighbors of a substation, permitted by filter values"""
        # This is clunky but allows us to not re-implement the logic of filters in multiple places
        valid_edges = PowerAssetList(
            [e[2]["object"] for e in self._power_graph.graph.edges(substation, data=True)],
            cognite_client=self._cognite_client,
        ).filter(grid_type=grid_type, base_voltage=base_voltage)
        edges = [
            edge for edge in self._power_graph.graph.edges(substation, data=True) if edge[2]["object"] in valid_edges
        ]
        neighbors = [n for e in edges for n in e[:2] if n != substation]
        return neighbors

    def expand_area(self, level=1, grid_type: str = None, base_voltage: Iterable = None) -> "PowerArea":
        """Expand the area by following line segments `level` times."""
        visited_nodes = set(self._graph.nodes)
        level_nodes = visited_nodes
        for _ in range(level):
            level_nodes = {
                nb
                for n in level_nodes
                for nb in self._find_neighbors(n, grid_type, base_voltage)
                if nb not in visited_nodes
            }
            visited_nodes |= level_nodes
        return PowerArea(self._cognite_client, [node for node in visited_nodes], self._power_graph)

    def is_connected(self) -> bool:
        return nx.is_connected(self._graph)

    def draw_with_map(self):
        """
        Plots substations and ac line segments overlayed on a world map
        """
        return PowerPlot.draw_with_map(self)

    def draw(self, labels="fixed", position="kamada", height=None):
        """Plots substations and ac line segments.

        Args:
            labels: 'fixed' to label by name, otherwise only shown on hovering over node.
            position: `source` to take positions from the assets xPosition/yPosition.
                      `project` to take positions from source and project them to meters east/north.
                      `spring` for a networkx spring location.
                      `kamada` for a networkx kamada-kawai location.
            height: figure height (width is set based on fixed aspect ratio)
        """
        return PowerPlot.draw(self, labels, position, height)

    def draw_flow(
        self,
        labels="fixed",
        position="kamada",
        height=None,
        timeseries_type="estimated_value",
        granularity="1h",
        date: "np.datetime64" = None,
    ):
        """Plots flow in area.

        Args:
            labels,position,height: as in `draw`
            timeseries_type: type of time series to retrieve, i.e. value/estimated_value.
            granularity: time step at which to average values over, as in the Python SDK `retrieve_dataframe` function.
            date: datetime object at which to visualize flow, use None for now.
        """
        return PowerPlot.draw_flow(
            self,
            labels=labels,
            position=position,
            height=height,
            timeseries_type=timeseries_type,
            granularity=granularity,
            date=date,
        )
