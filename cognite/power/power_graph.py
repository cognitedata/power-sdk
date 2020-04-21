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
        for rel in self._cognite_client.relationships.list(
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

        for rel in self._cognite_client.relationships.list(
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
                    ac_line_segment_con_substations[a].append((substation, terminal))

        self.graph = nx.Graph()
        self.graph.add_edges_from(
            (
                substation_from_extid[substation_from].name,
                substation_from_extid[substation_to].name,
                {
                    "object": ac_line_segment_from_extid[line],
                    "terminals": {  # Note that only one of the edges (a,b) and (b,a) is actually added, so this can not be by order
                        substation_from_extid[substation_from].name: terminal_from_extid[terminal_from],
                        substation_from_extid[substation_to].name: terminal_from_extid[terminal_to],
                    },
                },
            )
            for substation_from, acls in substation_con_ac_line_segments.items()
            for line, terminal_from in acls
            for substation_to, terminal_to in ac_line_segment_con_substations[line]
            if substation_from != substation_to
        )
        self.graph.add_nodes_from((substation.name, {"object": substation}) for substation in substations)
