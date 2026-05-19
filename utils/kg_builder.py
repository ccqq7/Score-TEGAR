from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.label_config import filter_kg_payload


UNDIRECTED_RELATIONS = {"often_cooccur", "statistical_exclusion"}
DIRECTED_RELATIONS = {"hierarchical"}


class KGDefinition:
    def __init__(self, kg_path: str | Path):
        kg_path = Path(kg_path)
        with kg_path.open("r", encoding="utf-8") as f:
            payload = filter_kg_payload(json.load(f))

        self.kg_path = kg_path.resolve()
        self.payload = payload
        self.nodes: list[str] = payload["nodes"]
        self.node_count: int = payload["node_count"]
        self.edges: dict[str, list[list[str]]] = payload["edges"]
        self.label_to_index: dict[str, int] = payload["label_to_index"]
        self.label_frequency: dict[str, int] = payload["label_frequency"]

    def get_adjacency_matrices(
        self,
        sparse: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        matrices: dict[str, torch.Tensor] = {}
        for relation_name, edge_list in self.edges.items():
            dense = torch.zeros((self.node_count, self.node_count), dtype=dtype, device=device)
            for src_label, dst_label in edge_list:
                src = self.label_to_index[src_label]
                dst = self.label_to_index[dst_label]
                dense[src, dst] = 1.0
                if relation_name in UNDIRECTED_RELATIONS:
                    dense[dst, src] = 1.0
            matrices[relation_name] = dense.to_sparse() if sparse else dense
        return matrices

    def get_homogeneous_adjacency(
        self,
        sparse: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        dense = torch.zeros((self.node_count, self.node_count), dtype=dtype, device=device)
        for relation_name, edge_list in self.edges.items():
            for src_label, dst_label in edge_list:
                src = self.label_to_index[src_label]
                dst = self.label_to_index[dst_label]
                dense[src, dst] = 1.0
                dense[dst, src] = 1.0
            if relation_name in DIRECTED_RELATIONS:
                dense = torch.maximum(dense, dense.t())
        return dense.to_sparse() if sparse else dense


def get_adjacency_matrices(kg_path: str | Path, sparse: bool = True) -> dict[str, torch.Tensor]:
    return KGDefinition(kg_path).get_adjacency_matrices(sparse=sparse)
