"""Exact directed reachability using the strongly connected condensation DAG."""
from collections import defaultdict
import networkx as nx


class DirectedReachability:
    def __init__(self, graph):
        self.component = {}
        for i, nodes in enumerate(nx.strongly_connected_components(graph)):
            for node in nodes:
                self.component[node] = i
        self.successors = defaultdict(set)
        for u, v in graph.edges():
            a, b = self.component[u], self.component[v]
            if a != b:
                self.successors[a].add(b)
        self.descendants = {}

    def __call__(self, source, target):
        a, b = self.component[source], self.component[target]
        if a == b:
            return True
        if a not in self.descendants:
            seen = {a}
            stack = [a]
            while stack:
                current = stack.pop()
                for nxt in self.successors[current]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            self.descendants[a] = seen
        return b in self.descendants[a]
