from collections import defaultdict

import networkx as nx
import textdistance

from cognite.power.data_classes import PowerAsset


class PowerGraph:
    def __init__(self, cognite_client):
        """Initializes a power graph. An instance of this is created when creating the first PowerArea, it should not be instantiated elsewhere."""
        self._cognite_client = cognite_client
        self._load()

    def helpful_substation_lookup(self, substation: str):
        if substation in self.graph.nodes:
            return substation, self.graph.nodes(data=True)[substation]
        else:
            close_matches = [name for name in list(self.graph.nodes) if textdistance.hamming(substation, name) <= 1]
            helper_message = " Did you mean: {}?".format(close_matches) if len(close_matches) > 0 else ""
            raise KeyError("Did not find substation '{}'.{}".format(substation, helper_message))

    def _load(self):
        substations = self._cognite_client.substations.list()
        ac_line_segments = self._cognite_client.ac_line_segments.list()

        substation_from_extid = {s.external_id: s for s in substations if s.external_id}
        ac_line_segment_from_extid = {s.external_id: s for s in ac_line_segments if s.external_id}

        terminal_con_ac_line_segments = defaultdict(list)
        for rel in self._cognite_client.relationships_playground.list(
            targets=[{"resourceId": acl.external_id} for acl in ac_line_segments],
            relationship_type="connectsTo",
            limit=None,
        ):
            terminal_con_ac_line_segments[rel.source["resourceId"]].append(rel.target["resourceId"])

        terminals = self._cognite_client.assets.retrieve_multiple(
            external_ids=list(terminal_con_ac_line_segments.keys())
        )
        terminal_from_extid = {}
        for check_terminal in terminals:
            if check_terminal.metadata.get("type", None) != "Terminal":
                del terminal_con_ac_line_segments[check_terminal.external_id]
            else:
                terminal_from_extid[check_terminal.external_id] = PowerAsset._load_from_asset(
                    check_terminal, "Terminal", cognite_client=self._cognite_client
                )

        substation_con_ac_line_segments = defaultdict(list)
        ac_line_segment_con_substations = defaultdict(list)

        for rel in self._cognite_client.relationships_playground.list(
            targets=[{"resourceId": s.external_id} for s in substations], relationship_type="belongsTo", limit=None
        ):
            substation = rel.target["resourceId"]
            terminal = rel.source["resourceId"]
            if substation in substation_from_extid and terminal in terminal_con_ac_line_segments:
                ac_line_segments = terminal_con_ac_line_segments.get(terminal, [])
                substation_con_ac_line_segments[substation].extend(
                    [(line_segment, terminal) for line_segment in ac_line_segments]
                )
                for a in ac_line_segments:
                    ac_line_segment_con_substations[a].append({"substation": substation, "terminal": terminal})

        self.graph = nx.MultiGraph()

        edges = (
            (
                substation_from_extid[data[0]["substation"]].name,
                substation_from_extid[data[1]["substation"]].name,
                {
                    "object": ac_line_segment_from_extid[acls],
                    "terminals": {
                        substation_from_extid[data[0]["substation"]].name: terminal_from_extid[data[0]["terminal"]],
                        substation_from_extid[data[1]["substation"]].name: terminal_from_extid[data[1]["terminal"]],
                    },
                },
            )
            for acls, data in ac_line_segment_con_substations.items()
            if len(data) == 2  # Skipping dangling line segments
            and data[0]["substation"] != data[1]["substation"]  # Skipping self-loops
        )
        self.graph.add_edges_from(edges)
        self.graph.add_nodes_from((substation.name, {"object": substation}) for substation in substations)
