from __future__ import annotations

import torch

from build_semantic_affinity import build_semantic_affinity_edges


def test_unseen_semantic_neighbors_are_seen_only_and_annotation_independent() -> None:
    labels = ["seen_a", "seen_b", "seen_c", "unseen_x", "unseen_y"]
    anchors = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.95, 0.05],
            [0.05, 0.95],
        ]
    )
    seen = labels[:3]
    unseen = labels[3:]
    edges = build_semantic_affinity_edges(
        label_names=labels,
        anchors=anchors,
        seen_labels=seen,
        unseen_labels=unseen,
        top_k=1,
    )
    edge_sets = [set(edge) for edge in edges]
    for unseen_label in unseen:
        incident = [edge for edge in edge_sets if unseen_label in edge]
        assert len(incident) == 1
        other = next(iter(incident[0] - {unseen_label}))
        assert other in seen
    assert not any(set(unseen).issubset(edge) for edge in edge_sets)
