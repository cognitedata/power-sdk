import math
from collections import defaultdict
from typing import List

import networkx as nx

from cognite.client.data_classes import Asset
from cognite.power import PowerAsset, PowerAssetList


class PowerGraph:
    def __init__(self, cognite_client, subgraph_nodes: List[str] = None, full_graph=None):
        self._cognite_client = cognite_client
        if subgraph_nodes:
            self.full_graph = full_graph
            self.G = nx.Graph()
            all_nodes = full_graph.nodes(data=True)
            self.G.add_nodes_from((k, all_nodes[k]) for k in subgraph_nodes)
            self.G.add_edges_from(
                (n, nbr, d)
                for n, nbrs in full_graph.adj.items()
                if n in subgraph_nodes
                for nbr, d in nbrs.items()
                if nbr in subgraph_nodes
            )
        else:
            self.load()
            self.full_graph = self.G

    def load(self):
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

        self.G = nx.Graph()
        self.G.add_edges_from(
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
        self.G.add_nodes_from((substation.name, {"object": substation}) for substation in substations)

    @staticmethod
    def _graph_ac_line_segments(graph, client):
        return PowerAssetList([obj for f, t, obj in graph.edges(data="object")], cognite_client=client)

    @staticmethod
    def _graph_substations(graph, client):
        return PowerAssetList([obj for n, obj in graph.nodes(data="object")], cognite_client=client)

    def substations(self):
        return self._graph_substations(self.G, self._cognite_client)

    def ac_line_segments(self):
        return self._graph_ac_line_segments(self.G, self._cognite_client)

    def _node_locations(self):
        node_loc = {
            name: [
                float(substation.metadata.get("PositionPoint.xPosition", math.nan)),
                float(substation.metadata.get("PositionPoint.yPosition", math.nan)),
            ]
            for name, substation in self.G.nodes(data="object")
        }
        orphan_count = 0
        for it in range(2):
            for s, loc in node_loc.items():
                if math.isnan(loc[0]):
                    nb_locs = [node_loc[n] for n in nx.neighbors(self.G, s) if not math.isnan(node_loc[n][0])]
                    mean_loc = [sum(c) / len(nb_locs) for c in zip(*nb_locs)]
                    if len(mean_loc) == 2:
                        node_loc[s] = mean_loc
                    elif it == 1:
                        node_loc[s] = [20, 55 + orphan_count]  # TODO don't hardcode this
                        orphan_count += 1
        return node_loc

    def _subgraph(self, nodes):
        nodes = [n.name if isinstance(n, (Asset)) else n for n in nodes]
        return PowerGraph(self._cognite_client, nodes, self.full_graph)

    def select_region(self, nodes):
        return self._subgraph(nodes)

    def expand_region(self, level=1):
        level_nodes = self.G.nodes
        visited_nodes = set(level_nodes)
        for _ in range(level):
            level_nodes = {
                nb for n in level_nodes for nb in nx.neighbors(self.full_graph, n) if nb not in visited_nodes
            }
            visited_nodes |= level_nodes
        return self._subgraph(visited_nodes)

    def interface(self):
        interface_edges = [(f, t) for f, t in sg.full_graph.edges if (f in sg.G) ^ (t in sg.G)]
        edge_data = self.full_graph.edges(data="object")
        return [obj for f, t, obj in edge_data if (f, t) in interface_edges]

    def draw(self, labels="name", **kwargs):
        args = {"font_size": 25, "pos": self._node_locations()}
        if labels and labels not in kwargs:
            args["labels"] = {n: n for n in self.G.nodes}
        nx.draw(self.G, **{**args, **kwargs})
