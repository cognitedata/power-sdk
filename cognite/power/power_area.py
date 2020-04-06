import math
from collections import defaultdict
from typing import *

import networkx as nx
import plotly.graph_objs as go
from matplotlib.colors import LinearSegmentedColormap

from cognite.client.data_classes import Asset
from cognite.power.data_classes import ACLineSegment, PowerAssetList, Substation


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
        self._graph = nx.Graph()
        all_nodes = self._power_graph.graph.nodes(data=True)
        self._graph.add_nodes_from((k, all_nodes[k]) for k in substations)
        self._graph.add_edges_from(
            (n, nbr, d)
            for n, nbrs in self._power_graph.graph.adj.items()
            if n in substations
            for nbr, d in nbrs.items()
            if nbr in substations
        )

    @classmethod
    def from_interface(
        cls,
        cognite_client,
        power_graph,
        ac_line_segments: List[ACLineSegment],
        interior_station: Union[Substation, str],
        base_voltage: Iterable = None,
    ):
        """Creates a power area from a list of ac line segments, interpreted as the interface of the area, as well as an interior substation"""
        interior_station = interior_station.name if isinstance(interior_station, Substation) else interior_station
        interface_edges = [[s.name for s in acls.substations()] for acls in ac_line_segments]
        temp_graph = power_graph.graph.copy()
        for edge in interface_edges:
            temp_graph.remove_edge(edge[0], edge[1])
        nodes = PowerArea._expand_area(
            temp_graph, nodes=[interior_station], level=len(temp_graph), base_voltage=base_voltage
        )
        for edge in interface_edges:
            if edge[0] in nodes and edge[1] in nodes:
                raise ValueError("Inconsistent interface. The interface must create an isolated zone.")
        return cls(cognite_client, [n for n in nodes], power_graph)

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

    def interface(self, base_voltage: Iterable = None) -> PowerAssetList:
        """Return the list of ACLineSegments going in/out of the area."""

        # select edges with either from or to in graph but not both
        interface_edges = [
            acl
            for f, t, acl in self._power_graph.graph.edges(data="object")
            if (f in self._graph) ^ (t in self._graph) and (base_voltage is None or acl.base_voltage in base_voltage)
        ]
        return PowerAssetList(interface_edges, cognite_client=self._cognite_client)

    def _node_locations(self):
        node_loc = {
            name: [
                float(substation.metadata.get("PositionPoint.xPosition", math.nan)),
                float(substation.metadata.get("PositionPoint.yPosition", math.nan)),
            ]
            for name, substation in self._graph.nodes(data="object")
        }
        orphan_count = 0
        for it in range(2):
            for s, loc in node_loc.items():
                if math.isnan(loc[0]):
                    nb_locs = [node_loc[n] for n in nx.neighbors(self._graph, s) if not math.isnan(node_loc[n][0])]
                    mean_loc = [sum(c) / len(nb_locs) for c in zip(*nb_locs)]
                    if len(mean_loc) == 2:
                        node_loc[s] = mean_loc
                    elif it == 1:
                        node_loc[s] = [20, 55 + orphan_count]  # TODO don't hardcode this
                        orphan_count += 1
        return node_loc

    def draw(self, labels="fixed", position="source"):
        """Plots the graph.

        Args:
            labels: 'fixed' to label by name, otherwise only shown on hovering over node.
            position: `source` to take positions from the assets xPosition/yPosition. `spring` for a networkx spring location.
        """
        cmap = LinearSegmentedColormap.from_list("custom blue", ["#ffff00", "#002266"], N=12)

        def voltage_color(bv):
            c = ",".join(map(str, cmap(bv / 500)))
            return f"rgba({c})"

        if position == "source":
            node_positions = self._node_locations()
        elif position == "spring":
            node_positions = nx.spring_layout(self._graph)
        else:
            raise ValueError(f"Unknown layout {position}")

        node_plot_mode = "markers"
        if labels == "fixed":
            node_plot_mode = "markers+text"

        # plot each base voltage
        edges_by_bv = defaultdict(list)
        for f, t, obj in self._graph.edges(data="object"):
            edges_by_bv[obj.base_voltage].append((f, t, obj))

        edge_traces = []
        hidden_edge_nodes = []
        edge_lbl = []
        for base_voltage, edge_list in edges_by_bv.items():
            bv_edge_points = sum([[node_positions[f], node_positions[t], [None, None]] for f, t, obj in edge_list], [])
            hidden_edge_nodes.extend(
                [(node_positions[f][0] + node_positions[t][0]) / 2, (node_positions[f][1] + node_positions[t][1]) / 2]
                for f, t, obj in edge_list
            )
            edge_lbl.extend(f"{obj.name}: {obj.base_voltage} kV" for f, t, obj in edge_list)
            edge_x, edge_y = zip(*bv_edge_points)
            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=1.5, color=voltage_color(base_voltage)),
                    hoverinfo="none",
                    mode="lines",
                )
            )

        edge_node_x, edge_node_y = zip(*hidden_edge_nodes)
        edge_node_trace = go.Scatter(
            x=edge_node_x, y=edge_node_y, text=edge_lbl, mode="markers", hoverinfo="text", marker=dict(size=0.001)
        )

        node_x, node_y = zip(*[xy for lbl, xy in node_positions.items()])
        node_lbl = [lbl for lbl, xy in node_positions.items()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_lbl,
            mode=node_plot_mode,
            textposition="top center",
            hoverinfo="text",
            marker=dict(size=15, line_width=2, color="orangered"),
        )

        fig = go.Figure(
            data=edge_traces + [edge_node_trace, node_trace],
            layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        return fig

    @staticmethod
    def _expand_area(networkx_graph, nodes, level, base_voltage):
        visited_nodes = set(nodes)
        level_nodes = nodes
        for _ in range(level):
            level_nodes = {
                nb
                for n in level_nodes
                for nb, data in networkx_graph[n].items()
                if nb not in visited_nodes and (base_voltage is None or data["object"].base_voltage in base_voltage)
            }
            visited_nodes |= level_nodes
        return visited_nodes

    def expand_area(self, level=1, base_voltage: Iterable = None) -> "PowerArea":
        """Expand the area by following line segments `level` times."""
        nodes = self._expand_area(self._power_graph.graph, self._graph.nodes, level, base_voltage)
        return PowerArea(self._cognite_client, [node for node in nodes], self._power_graph)

    def is_connected(self) -> bool:
        return nx.is_connected(self._graph)
