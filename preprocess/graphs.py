# Stage 3: Build the route graph and its district subgraphs, for use in graph-based models.

import numpy as np
import pandas as pd
import networkx as nx
from . import config as C
from . import raw, sensor_selection


def _directed_edges(graph, nodes: set[int]) -> pd.DataFrame:
    """
    Return a DataFrame of directed edges between the given nodes
    """
    sub = graph.subgraph(nodes)
    best = {}

    # Go through the edges and keep the best (highest weight) edge for each directed pair.
    for u, v, d in sub.edges(data=True):
        u, v = int(u), int(v)

        # no self-loops
        if u == v:
            continue

        w, dist = float(d["weight"]), float(d["distance"])
        cur = best.get((u, v))
        if cur is None or w > cur[0]:
            best[(u, v)] = (w, dist)

    rows = [(u, v, w, dist) for (u, v), (w, dist) in best.items()]
    df = pd.DataFrame(rows, columns=["source", "target", "weight", "distance"])

    return df.sort_values(["source", "target"], ignore_index=True)

def _attach_indices(edges: pd.DataFrame, sensor_ids: np.ndarray, districts: pd.Series) -> pd.DataFrame:
    """
    Attach array indices to the edge DataFrame.
    """
    idx = pd.Index(sensor_ids)
    edges = edges.copy()
    edges["source_index"] = idx.get_indexer(edges["source"])
    edges["target_index"] = idx.get_indexer(edges["target"])

    if (edges[["source_index", "target_index"]] < 0).to_numpy().any():
        raise ValueError("edge endpoint outside the selected sensor set")
    
    edges["source_district"] = edges["source"].map(districts).astype("Int64")
    edges["target_district"] = edges["target"].map(districts).astype("Int64")
    edges["cross_district"] = (edges["source_district"] != edges["target_district"])

    return edges


def build(out_dir=C.OUT_DIR, verbose=True) -> dict:
    """
    Build the directed route graph and its district subgraphs
    """

    # load route graph and selected sensors
    graph = raw.load_route_graph()
    inc = sensor_selection.included(out_dir)
    sensor_ids = inc["sensor_id"].to_numpy(dtype=np.int64)
    districts = pd.Series(inc["district"].to_numpy(), index=sensor_ids)

    gdir = out_dir / "graph"
    gdir.mkdir(parents=True, exist_ok=True)

    # remove any existing district subgraph files
    for stale in list(gdir.glob("edges_district_*.csv")) + list(gdir.glob("nodes_district_*.csv")):
        stale.unlink()

    # Full directed graph
    edges = _attach_indices(_directed_edges(graph, set(sensor_ids.tolist())), sensor_ids, districts)
    within = edges[~edges["cross_district"]].reset_index(drop=True)
    cross = edges[edges["cross_district"]].reset_index(drop=True)

    edges.to_csv(gdir / "edges_full.csv", index=False)
    cross.to_csv(gdir / "edges_cross_district.csv", index=False)

    # node list
    pd.DataFrame({
        "array_index": np.arange(len(sensor_ids)),
        "sensor_id": sensor_ids,
        "district": inc["district"].to_numpy(),
        "latitude": inc["latitude"].to_numpy(),
        "longitude": inc["longitude"].to_numpy(),
    }).to_csv(gdir / "nodes.csv", index=False)

    # district subgraphs
    per_district = {}
    for d, grp in inc.groupby("district", dropna=True):
        d = int(d)
        d_ids = grp["sensor_id"].to_numpy(dtype=np.int64)
        d_edges = _directed_edges(graph, set(d_ids.tolist()))

        # each district gets own local numbering of nodes
        local = pd.Index(d_ids)
        d_edges["source_index"] = local.get_indexer(d_edges["source"])
        d_edges["target_index"] = local.get_indexer(d_edges["target"])
        d_edges.to_csv(gdir / f"edges_district_{d:02d}.csv", index=False)

        # write node list for district
        gidx = pd.Index(sensor_ids).get_indexer(d_ids)
        pd.DataFrame({
            "local_index": np.arange(len(d_ids)),
            "global_index": gidx,
            "sensor_id": d_ids,
        }).to_csv(gdir / f"nodes_district_{d:02d}.csv", index=False)
        
        per_district[d] = int(len(d_edges))

    # summary of full graph and district subgraphs
    und = nx.Graph()
    und.add_nodes_from(sensor_ids.tolist())
    und.add_edges_from(zip(within["source"], within["target"]))
    und.add_edges_from(zip(cross["source"], cross["target"]))
    comps = sorted((len(c) for c in nx.connected_components(und)), reverse=True)

    summary = {
        "n_nodes": int(len(sensor_ids)),
        "n_edges_full": int(len(edges)),
        "n_edges_within_district": int(len(within)),
        "n_edges_cross_district": int(len(cross)),
        "undirected_connected": bool(len(comps) == 1),
        "undirected_component_sizes": comps[:10],
        "n_isolated_nodes": int(sum(1 for n in und.nodes if und.degree(n) == 0)),
        "district_edges": {str(k): v for k, v in sorted(per_district.items())},
    }

    C.write_json(summary, C.reports_dir(out_dir) / "graph_summary.json")
    
    if verbose:
        print(f"  {summary['n_nodes']} nodes, {summary['n_edges_full']:,} edges "
              f"({summary['n_edges_cross_district']:,} cross-district), "
              f"connected={summary['undirected_connected']}")
        
    return summary


def load_full_edges(out_dir=C.OUT_DIR) -> pd.DataFrame:
    return pd.read_csv(out_dir / "graph" / "edges_full.csv")


def load_district_edges(district: int, out_dir=C.OUT_DIR) -> pd.DataFrame:
    return pd.read_csv(out_dir / "graph" / f"edges_district_{int(district):02d}.csv")


def load_district_nodes(district: int, out_dir=C.OUT_DIR) -> pd.DataFrame:
    return pd.read_csv(out_dir / "graph" / f"nodes_district_{int(district):02d}.csv")


def load_cross_district_edges(out_dir=C.OUT_DIR) -> pd.DataFrame:
    return pd.read_csv(out_dir / "graph" / "edges_cross_district.csv")
