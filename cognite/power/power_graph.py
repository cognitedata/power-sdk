import networkx as nx
from matplotlib import pyplot as plt


class PowerGraph:
    def __init__(self):
        self.load()

    def load(self):
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
        self.G.add_edges_from(ss_edges)
