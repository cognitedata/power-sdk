from collections import defaultdict
from typing import *
import networkx as nx
from cognite.power.power_area import PowerArea


class PowerGraph:
    def __init__(self, cognite_client):
        """Initializes a power graph. An instance of this is created when creating a PowerClient, it should not be instantiated elsewhere."""
        self._cognite_client = cognite_client
        self._load()

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
