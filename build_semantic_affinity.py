from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


EMPIRICAL_RELATIONS = {"often_cooccur", "statistical_exclusion", "hierarchical"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add an annotation-independent semantic-affinity relation to a strict train-seen graph."
    )
    parser.add_argument("--kg_path", required=True)
    parser.add_argument("--text_anchors", required=True)
    parser.add_argument("--outer_split", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def build_semantic_affinity_edges(
    *,
    label_names: list[str],
    anchors: torch.Tensor,
    seen_labels: list[str],
    unseen_labels: list[str],
    top_k: int,
) -> list[list[str]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if anchors.ndim != 2 or anchors.shape[0] != len(label_names):
        raise ValueError("Anchor tensor must have one row per label.")
    label_to_index = {label: index for index, label in enumerate(label_names)}
    missing = (set(seen_labels) | set(unseen_labels)) - set(label_names)
    if missing:
        raise ValueError(f"Split labels missing from anchors: {sorted(missing)}")
    if set(seen_labels) & set(unseen_labels):
        raise ValueError("seen_labels and unseen_labels must be disjoint")
    if set(seen_labels) | set(unseen_labels) != set(label_names):
        raise ValueError("Outer split must partition the full active label vocabulary.")

    normalized = F.normalize(anchors.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))
    edges: set[tuple[str, str]] = set()

    # Seen nodes use only other seen nodes as neighbors.  This prevents the
    # training-side graph topology from depending on which unseen labels happen
    # to be semantically close, while still giving seen propagation a text-only
    # structural prior.
    for source in seen_labels:
        candidates = [label for label in seen_labels if label != source]
        if not candidates:
            continue
        source_index = label_to_index[source]
        ranked = sorted(
            candidates,
            key=lambda label: (-float(similarity[source_index, label_to_index[label]]), label),
        )[: min(top_k, len(candidates))]
        edges.update(_canonical_edge(source, target) for target in ranked)

    # Every outer-unseen node is connected only to outer-seen nodes.  Text
    # anchor similarity is available without unseen annotations and is never
    # represented as empirical co-occurrence.
    for source in unseen_labels:
        source_index = label_to_index[source]
        ranked = sorted(
            seen_labels,
            key=lambda label: (-float(similarity[source_index, label_to_index[label]]), label),
        )[: min(top_k, len(seen_labels))]
        edges.update(_canonical_edge(source, target) for target in ranked)
    return [list(edge) for edge in sorted(edges)]


def augment_graph(
    *,
    kg_path: Path,
    anchor_path: Path,
    split_path: Path,
    top_k: int,
    output_path: Path,
) -> dict:
    graph = json.loads(kg_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    label_names = list(anchor_payload["label_names"])
    anchors = anchor_payload["anchors"].float()
    if label_names != list(graph["nodes"]):
        raise ValueError("Anchor label order must exactly match graph node order.")

    seen_labels = list(split["seen_labels"])
    unseen_labels = list(split["unseen_labels"])
    for relation in EMPIRICAL_RELATIONS:
        endpoints = {
            endpoint
            for edge in graph.get("edges", {}).get(relation, [])
            for endpoint in edge
        }
        leaked = sorted(endpoints - set(seen_labels))
        if leaked:
            raise ValueError(f"Empirical relation {relation} contains outer-unseen endpoints: {leaked}")

    semantic_edges = build_semantic_affinity_edges(
        label_names=label_names,
        anchors=anchors,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        top_k=top_k,
    )
    graph.setdefault("edges", {})["semantic_affinity"] = semantic_edges
    graph.setdefault("relation_provenance", {})["semantic_affinity"] = {
        "annotation_independent": True,
        "source": "basic_text_anchor_cosine",
        "top_k": int(top_k),
        "unseen_neighbor_scope": "outer_seen_only",
        "text_anchors_sha256": _sha256_file(anchor_path),
        "outer_split_sha256": _sha256_file(split_path),
    }
    graph["protocol_mode"] = "strict_train_seen_plus_text_affinity_v1"
    graph.setdefault("leakage_audit", {})["semantic_affinity_annotation_independent"] = True
    graph["leakage_audit"]["semantic_affinity_edge_count"] = len(semantic_edges)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def main() -> None:
    args = parse_args()
    augment_graph(
        kg_path=Path(args.kg_path).resolve(),
        anchor_path=Path(args.text_anchors).resolve(),
        split_path=Path(args.outer_split).resolve(),
        top_k=int(args.top_k),
        output_path=Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
