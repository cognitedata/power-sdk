import math
from collections import defaultdict
from typing import List, Union

import matplotlib
import networkx as nx
import plotly as py
import plotly.graph_objs as go
from matplotlib.colors import LinearSegmentedColormap

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
            node_positions = nx.spring_layout(self.graph)
        else:
            raise ValueError(f"Unknown layout {position}")

        node_plot_mode = "markers"
        if labels == "fixed":
            node_plot_mode = "markers+text"

        # plot each base voltage
        edges_by_bv = defaultdict(list)
        for f, t, obj in self.graph.edges(data="object"):
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
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        return fig
