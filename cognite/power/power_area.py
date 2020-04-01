import math
from typing import *

import networkx as nx

from cognite.power.data_classes import PowerAssetList, Substation


class PowerArea:
    """
    This is a good docstring.
    """

    def __init__(self, cognite_client, substations: List[Substation]):
        self._cognite_client = cognite_client
        if len(substations) < 1:
            raise ValueError("A power area must have at least one substation")
        full_graph = cognite_client.graph.graph
        self._graph = nx.Graph()
        all_nodes = full_graph.nodes(data=True)
        self._graph.add_nodes_from((k, all_nodes[k]) for k in substations)
        self._graph.add_edges_from(
            (n, nbr, d)
            for n, nbrs in full_graph.adj.items()
            if n in substations
            for nbr, d in nbrs.items()
            if nbr in substations
        )
        if not nx.is_connected(self._graph):
            raise ValueError("The supplied substations are not connected.")

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

    def interface(self) -> PowerAssetList:
        """Return the list of ACLineSegments on the edge of the selected region."""

        # select edges with either from or to in graph but not both
        interface_edges = [
            (f, t) for f, t in self._cognite_client.graph.graph.edges if (f in self._graph) ^ (t in self._graph)
        ]
        edge_data = self._cognite_client.graph.graph.edges(data="object")
        return PowerAssetList(
            [obj for f, t, obj in edge_data if (f, t) in interface_edges], cognite_client=self._cognite_client
        )

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

    def draw(self, labels="name", pos="cdf", label_args=None, **kwargs):
        """Plots the graph.

        Args:
            labels: 'name' to label by name, other values are passed to networkx.draw (e.g. `None`)
            pos: `cdf` to take positions from the assets xPosition/yPosition. `spring` for a networkx spring location. Other values passed to `networkx.draw` directly.
            label_args: passed to `networkx.draw_networkx_labels`, e.g. {'font_size':10}
            kwargs: are passed to `networkx.draw`
        """
        if pos == "cdf":
            pos = self._node_locations()
        elif pos == "spring":
            pos = nx.spring_layout(self._graph)

        if labels == "name":
            labels = {n: n for n in self._graph.nodes}

        draw_args = kwargs
        nx.draw(self._graph, pos=pos, **draw_args)
        if labels:
            offset = 0  # (max_y - min(ys)) / 50
            font_size = 20
            label_args = {"font_size": font_size, **(label_args or {})}
            offset_pos = {n: [xy[0], xy[1] + offset] for n, xy in pos.items()}
            nx.draw_networkx_labels(self._graph, pos=offset_pos, **label_args)

    def expand_area(self, level=1) -> "PowerArea":
        """Expand the area by following line segments `level` times."""
        level_nodes = self._graph.nodes
        visited_nodes = set(level_nodes)
        # TODO: this loop does not need to lock at all nodes for each iteration, but I do not want to optimize before we have tests in place
        for _ in range(level):
            level_nodes = {
                nb
                for n in level_nodes
                for nb in nx.neighbors(self._cognite_client.graph.graph, n)
                if nb not in visited_nodes
            }
            visited_nodes |= level_nodes
        return PowerArea(self._cognite_client, [node for node in visited_nodes])
