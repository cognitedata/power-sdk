import math

import networkx as nx

class PowerGraph:
    def __init__(self, client):
        self.load(client)

    def load(self, client):
        ssl = client.substations.list()
        acl = client.ac_line_segments.list()
        ss_extid = {s.external_id: s for s in ssl if s.external_id}
        acl_extid = {s.external_id: s for s in acl if s.external_id}

        terminal_map = defaultdict(list)
        for rel in client.relationships.list(
            targets=[{"resourceId": s.external_id} for s in acl], relationship_type="connectsTo", limit=None
        ):
            terminal = rel.source["resourceId"]
            acls = rel.target["resourceId"]
            terminal_map[terminal].append(acls)

        for check_terminal in client.assets.retrieve_multiple(external_ids=list(terminal_map.keys())):
            if check_terminal.metadata.get("type", None) != "Terminal":
                del terminal_map[check_terminal.external_id]

        ss_to_acls_map = defaultdict(list)
        acls_to_ss_map = defaultdict(list)

        for rel in client.relationships.list(
            targets=[{"resourceId": s.external_id} for s in ssl], relationship_type="belongsTo", limit=None
        ):
            ss = rel.target["resourceId"]
            terminal = rel.source["resourceId"]
            if ss in ss_extid and terminal in terminal_map:
                acls = terminal_map.get(terminal, [])
                ss_to_acls_map[ss].extend(acls)
                for a in acls:
                    acls_to_ss_map[a].append(ss)

        edges = [
            (sfrom, sto, a)
            for sfrom, acls in ss_to_acls_map.items()
            for a in acls
            for sto in acls_to_ss_map[a]
            if sfrom != sto
        ]
        ss_edges = [(ss_extid[f], ss_extid[t], {"Line": a}) for f, t, a in edges]

        self.G = nx.Graph()
        self.G.add_nodes_from(ss_extid.values())
        self.G.add_edges_from(ss_edges)

    def _node_locations(self):
        node_loc = {
            s: [
                float(s.metadata.get("PositionPoint.xPosition", math.nan)),
                float(s.metadata.get("PositionPoint.yPosition", math.nan)),
            ]
            for s in self.G.nodes
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

    def draw(self):
        nx.draw(self.G, labels={n: n.name for n in self.G.nodes()}, font_size=30, pos=self._node_locations())
