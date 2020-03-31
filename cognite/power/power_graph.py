import math
from collections import defaultdict
from typing import List, Union

import networkx as nx

from cognite.client.data_classes import Asset
from cognite.power.data_classes import PowerAsset, PowerAssetList, Substation


class PowerGraph:
    def __init__(self, cognite_client, subgraph_nodes: List[str] = None, full_graph=None):
        """Initializes a power graph, should typically be accessed via client.power_graph()"""
        self._cognite_client = cognite_client
        if subgraph_nodes:
            self.full_graph = full_graph
            self.graph = nx.Graph()
            all_nodes = full_graph.nodes(data=True)
            self.graph.add_nodes_from((k, all_nodes[k]) for k in subgraph_nodes)
            self.graph.add_edges_from(
                (n, nbr, d)
                for n, nbrs in full_graph.adj.items()
                if n in subgraph_nodes
                for nbr, d in nbrs.items()
                if nbr in subgraph_nodes
            )
        else:
            self._load()
            self.full_graph = self.graph

    def _load(self):
        substations = self._cognite_client.substations.list()
        ac_line_segments = self._cognite_client.ac_line_segments.list()

        substation_from_extid = {s.external_id: s for s in substations if s.external_id}
        ac_line_segment_from_extid = {s.external_id: s for s in ac_line_segments if s.external_id}

        terminal_con_ac_line_segments = defaultdict(list)
        for rel in self._cognite_client.relationships.list(
            targets=[{"resourceId": acl.external_id} for acl in ac_line_segments],
            relationship_type="connectsTo",
            limit=None,
        ):
            terminal_con_ac_line_segments[rel.source["resourceId"]].append(rel.target["resourceId"])

        for check_terminal in self._cognite_client.assets.retrieve_multiple(
            external_ids=list(terminal_con_ac_line_segments.keys())
        ):
            if check_terminal.metadata.get("type", None) != "Terminal":
                del terminal_con_ac_line_segments[check_terminal.external_id]

        substation_con_ac_line_segments = defaultdict(list)
        ac_line_segment_con_substations = defaultdict(list)

        for rel in self._cognite_client.relationships.list(
            targets=[{"resourceId": s.external_id} for s in substations], relationship_type="belongsTo", limit=None
        ):
            substation = rel.target["resourceId"]
            terminal = rel.source["resourceId"]
            if substation in substation_from_extid and terminal in terminal_con_ac_line_segments:
                ac_line_segments = terminal_con_ac_line_segments.get(terminal, [])
                substation_con_ac_line_segments[substation].extend(ac_line_segments)
                for a in ac_line_segments:
                    ac_line_segment_con_substations[a].append(substation)

        self.graph = nx.Graph()
        self.graph.add_edges_from(
            (
                substation_from_extid[substation_from].name,
                substation_from_extid[substation_to].name,
                {"object": ac_line_segment_from_extid[a]},
            )
            for substation_from, acls in substation_con_ac_line_segments.items()
            for a in acls
            for substation_to in ac_line_segment_con_substations[a]
            if substation_from != substation_to
        )
        self.graph.add_nodes_from((substation.name, {"object": substation}) for substation in substations)

    @staticmethod
    def _graph_ac_line_segments(graph, client):
        return PowerAssetList([obj for f, t, obj in graph.edges(data="object")], cognite_client=client)

    @staticmethod
    def _graph_substations(graph, client):
        return PowerAssetList([obj for n, obj in graph.nodes(data="object")], cognite_client=client)

    def substations(self) -> PowerAssetList:
        """Returns the list of Substations in the graph"""
        return self._graph_substations(self.graph, self._cognite_client)

    def ac_line_segments(self) -> PowerAssetList:
        """Returns the list of ACLineSegments in the graph"""
        return self._graph_ac_line_segments(self.graph, self._cognite_client)

    def _subgraph(self, nodes):
        nodes = [n.name if isinstance(n, (Asset)) else n for n in nodes]
        return PowerGraph(self._cognite_client, nodes, self.full_graph)

    def select_region(self, nodes: List[Union[Substation, str]]) -> "PowerGraph":
        """Select a region using a list of substations (or their names)"""
        return self._subgraph(nodes)

    def expand_region(self, level=1) -> "PowerGraph":
        """Expand the graph by following line segments `level` times."""
        level_nodes = self.graph.nodes
        visited_nodes = set(level_nodes)
        for _ in range(level):
            level_nodes = {
                nb for n in level_nodes for nb in nx.neighbors(self.full_graph, n) if nb not in visited_nodes
            }
            visited_nodes |= level_nodes
        return self._subgraph(visited_nodes)

    def interface(self) -> PowerAssetList:
        """Return the list of ACLineSegments on the edge of the selected region."""

        # select edges with either from or to in graph but not both
        interface_edges = [(f, t) for f, t in self.full_graph.edges if (f in self.graph) ^ (t in self.graph)]
        edge_data = self.full_graph.edges(data="object")
        return PowerAssetList(
            [obj for f, t, obj in edge_data if (f, t) in interface_edges], cognite_client=self._cognite_client
        )

    def _node_locations(self):
        node_loc = {
            name: [
                float(substation.metadata.get("PositionPoint.xPosition", math.nan)),
                float(substation.metadata.get("PositionPoint.yPosition", math.nan)),
            ]
            for name, substation in self.graph.nodes(data="object")
        }
        orphan_count = 0
        for it in range(2):
            for s, loc in node_loc.items():
                if math.isnan(loc[0]):
                    nb_locs = [node_loc[n] for n in nx.neighbors(self.graph, s) if not math.isnan(node_loc[n][0])]
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
            pos = nx.spring_layout(self.graph)

        if labels == "name":
            labels = {n: n for n in self.graph.nodes}

        draw_args = kwargs
        nx.draw(self.graph, pos=pos, **draw_args)
        if labels:
            ys = {xy[1] for n, xy in nx.spring_layout(self.graph).items()}
            max_y = max(ys)
            offset = 0  # (max_y - min(ys)) / 50
            font_size = 20
            label_args = {"font_size": font_size, **(label_args or {})}
            offset_pos = {n: [xy[0], xy[1] + offset] for n, xy in pos.items()}
            nx.draw_networkx_labels(self.graph, pos=offset_pos, **label_args)
