from typing import List, Set

import networkx as nx


def k_neighbors(G: nx.Graph, src: int, cutoff: int = 1):
    neighbors = {src}
    for _ in range(cutoff):
        for node in neighbors.copy():
            neighbors.update(G[node])
    return neighbors


def kth_neighbors(G: nx.Graph, src: int, kth: int = 1):
    neighbors: Set[int] = {src}
    neighbors_ltk = set()
    for _ in range(kth):
        neighbors_ltk.update(neighbors)
        neighbors = {m for node in neighbors for m in G[node]} - neighbors_ltk
    return neighbors


def try_merge_small(G: nx.Graph, groups: List[List[int]], min_cnt=3):
    if min_cnt <= 0:
        return groups, []

    small: List[int] = []
    large: List[List[int]] = []
    for group in groups:
        if len(group) < min_cnt:
            small.extend(group)
        else:
            large.append(group)
    if not small:
        return large, []

    maybe_merged: List[List[int]] = []
    for i, lg in enumerate(large):
        maybe_connected = lg + small
        subg = G.subgraph(maybe_connected)

        small = []
        for connected in map(list, nx.connected_components(subg)):
            if len(connected) >= min_cnt:
                maybe_merged.append(connected)
            else:
                small.extend(connected)
        if not small:
            maybe_merged += large[i + 1 :]
            break

    return maybe_merged, list(
        map(list, nx.connected_components(G.subgraph(small)))
    )
