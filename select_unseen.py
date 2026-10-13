from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.kg_builder import KGDefinition






def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic outer seen/unseen label splits.")
    parser.add_argument("--dataset", default="mlrsnet", choices=["mlrsnet", "bigearthnet", "chinamas"])
    parser.add_argument(
        "--label_names_file",
        default=None,
        help=(
            "Strict-mode label vocabulary (JSON list, JSON object with label_names/labels/nodes, "
            "one label per line, or a train feature-cache .pt containing label_names). "
            "Strict selection reads no KG, frequencies, annotation matrix, or text anchors."
        ),
    )
    parser.add_argument("--kg_path", default="./data/kg_definition.json")
    parser.add_argument("--text_anchors", default="./data/text_anchors.pt")
    parser.add_argument("--n_unseen", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--min_seen_neighbors", type=int, default=2)
    parser.add_argument("--min_semantic_neighbors", type=int, default=2)
    parser.add_argument("--semantic_threshold", type=float, default=0.3)
    parser.add_argument("--protected_top_k", type=int, default=10)
    parser.add_argument(
        "--legacy-selection",
        action="store_true",
        help="Explicitly enable the legacy KG/frequency/text-similarity-conditioned selector.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_dir", default=None)
    if argv is not None:
        return parser.parse_args(argv)
    return parser.parse_args()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_names(label_names_file: str | Path) -> tuple[list[str], dict[str, object]]:
    path = Path(label_names_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Label vocabulary file not found: {path}")

    source: str | None = None
    if path.suffix.lower() == ".pt":
        # mmap avoids materialising large cached tensors; strict split selection only
        # consults the split marker and label_names metadata below.
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Feature cache must contain a dictionary payload: {path}")
        if payload.get("split") != "train":
            raise ValueError(f"Feature cache must declare split='train': {path}")
        raw_labels = payload.get("label_names")
        if not isinstance(raw_labels, (list, tuple)):
            raise ValueError(f"Train feature cache must contain list-like label_names metadata: {path}")
        source = "train_feature_cache_label_names"
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_labels = payload
        elif isinstance(payload, dict):
            raw_labels = None
            for key in ("label_names", "labels", "nodes"):
                if isinstance(payload.get(key), list):
                    raw_labels = payload[key]
                    break
            if raw_labels is None and isinstance(payload.get("seen_labels"), list) and isinstance(payload.get("unseen_labels"), list):
                raw_labels = list(payload["seen_labels"]) + list(payload["unseen_labels"])
            if raw_labels is None:
                raise ValueError(
                    f"JSON vocabulary must contain label_names, labels, nodes, or seen_labels+unseen_labels: {path}"
                )
        else:
            raise ValueError(f"Unsupported label vocabulary payload in {path}")
    else:
        raw_labels = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    labels = [str(label).strip() for label in raw_labels if str(label).strip()]
    if not labels:
        raise ValueError(f"Label vocabulary is empty: {path}")
    if len(set(labels)) != len(labels):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"Label vocabulary contains duplicates: {duplicates}")
    provenance = {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "label_names_sha256": _canonical_json_sha256(labels),
    }
    if source is not None:
        provenance["source"] = source
    return labels, provenance


def select_unseen_labels(
    label_names: list[str],
    n_unseen: int = 15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    """Uniform outer-label split independent of annotations and learned/model-derived structure."""
    labels = [str(label) for label in label_names]
    if len(set(labels)) != len(labels):
        raise ValueError("label_names must be unique")
    if n_unseen <= 0 or n_unseen >= len(labels):
        raise ValueError(f"n_unseen must be in [1, {len(labels) - 1}], got {n_unseen}")

    # Sampling from a sorted copy makes the split invariant to incidental source-file ordering.
    sampling_labels = sorted(labels)
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.choice(len(sampling_labels), size=int(n_unseen), replace=False)
    unseen_set = {sampling_labels[int(index)] for index in sampled_indices.tolist()}
    seen = [label for label in labels if label not in unseen_set]
    unseen = [label for label in labels if label in unseen_set]
    unseen_info = [
        {
            "label": label,
            "selection_policy": "uniform_without_replacement",
            "sampling_index": int(sampling_labels.index(label)),
        }
        for label in unseen
    ]
    return seen, unseen, unseen_info


def load_kg(kg_path: str | Path) -> dict:
    return KGDefinition(kg_path).payload


def load_text_anchors(text_anchor_path: str | Path, label_names: list[str]) -> torch.Tensor:
    payload = torch.load(Path(text_anchor_path).resolve(), map_location="cpu", weights_only=False)
    anchors = payload["anchors"].float()
    anchor_labels = list(payload["label_names"])
    if anchor_labels != label_names:
        raise ValueError("text_anchors.pt label order does not match kg_definition.json nodes")
    return F.normalize(anchors, dim=-1)


def build_graph_stats(kg_definition: dict) -> tuple[dict[str, int], dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    nodes = list(kg_definition["nodes"])
    degree = {node: 0 for node in nodes}
    neighbors = {node: set() for node in nodes}
    relation_neighbors = {
        node: {"often_cooccur": set(), "statistical_exclusion": set(), "hierarchical": set()} for node in nodes
    }
    for relation_name, edges in kg_definition["edges"].items():
        for src, dst in edges:
            if src in degree:
                degree[src] += 1
                neighbors[src].add(dst)
                relation_neighbors[src][relation_name].add(dst)
            if dst in degree:
                degree[dst] += 1
                neighbors[dst].add(src)
                relation_neighbors[dst][relation_name].add(src)
    return degree, neighbors, relation_neighbors


def _semantic_seen_neighbors(
    label: str,
    nodes: list[str],
    label_to_index: dict[str, int],
    similarity: torch.Tensor,
    excluded: set[str],
    threshold: float,
) -> list[tuple[str, float]]:
    idx = label_to_index[label]
    pairs: list[tuple[str, float]] = []
    for other_idx, other_label in enumerate(nodes):
        if other_label == label or other_label in excluded:
            continue
        score = float(similarity[idx, other_idx].item())
        if score >= threshold:
            pairs.append((other_label, score))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


def select_unseen_labels_legacy(
    kg_definition: dict,
    text_anchors: torch.Tensor,
    n_unseen: int = 15,
    seed: int = 42,
    min_seen_neighbors: int = 2,
    min_semantic_neighbors: int = 2,
    semantic_threshold: float = 0.3,
    protected_top_k: int = 10,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    nodes = list(kg_definition["nodes"])
    label_to_index = {label: idx for idx, label in enumerate(nodes)}
    freq = dict(kg_definition["label_frequency"])
    degree, neighbors, relation_neighbors = build_graph_stats(kg_definition)
    similarity = text_anchors @ text_anchors.T

    sorted_by_freq = sorted(nodes, key=lambda label: freq[label], reverse=True)
    protected = set(sorted_by_freq[: max(0, min(int(protected_top_k), len(sorted_by_freq)))])
    candidates = [
        label
        for label in nodes
        if label not in protected
        and degree[label] >= 2
        and len(
            _semantic_seen_neighbors(
                label=label,
                nodes=nodes,
                label_to_index=label_to_index,
                similarity=similarity,
                excluded={label},
                threshold=semantic_threshold,
            )
        )
        >= min_semantic_neighbors
    ]
    candidates = sorted(candidates, key=lambda label: (freq[label], label))
    rng = np.random.default_rng(seed)

    def seen_neighbor_count(label: str, current_unseen: set[str]) -> int:
        return sum(1 for neighbor in neighbors[label] if neighbor not in current_unseen)

    def is_valid_unseen(label: str, current_unseen: set[str]) -> bool:
        candidate_unseen = set(current_unseen)
        candidate_unseen.add(label)
        for eval_label in candidate_unseen:
            semantic_seen = _semantic_seen_neighbors(
                label=eval_label,
                nodes=nodes,
                label_to_index=label_to_index,
                similarity=similarity,
                excluded=candidate_unseen,
                threshold=semantic_threshold,
            )
            if seen_neighbor_count(eval_label, candidate_unseen) < min_seen_neighbors:
                return False
            if len(semantic_seen) < min_semantic_neighbors:
                return False
        return True

    def sample_interval_candidates_once() -> list[str] | None:
        if len(candidates) <= n_unseen:
            sampled = []
            for candidate_label in candidates:
                if is_valid_unseen(candidate_label, set(sampled)):
                    sampled.append(candidate_label)
            return sampled if len(sampled) >= n_unseen else None

        sampled = []
        used = set()
        positions = np.linspace(0, len(candidates) - 1, n_unseen)
        window_radius = max(1, len(candidates) // max(8, n_unseen * 2))
        for pos in positions:
            center = int(round(float(pos)))
            left = max(0, center - window_radius)
            right = min(len(candidates), center + window_radius + 1)
            current_unseen = set(sampled)
            pool = [
                candidate_label
                for candidate_label in candidates[left:right]
                if candidate_label not in used and is_valid_unseen(candidate_label, current_unseen)
            ]
            if not pool:
                pool = [
                    candidate_label
                    for candidate_label in candidates
                    if candidate_label not in used and is_valid_unseen(candidate_label, current_unseen)
                ]
            if not pool:
                return None
            choice = pool[int(rng.integers(0, len(pool)))]
            sampled.append(choice)
            used.add(choice)
        return sampled

    def sample_with_backtracking() -> list[str]:
        candidate_order = list(candidates)
        rng.shuffle(candidate_order)

        def dfs(start_idx: int, current_unseen: list[str]) -> list[str] | None:
            if len(current_unseen) >= n_unseen:
                return list(current_unseen[:n_unseen])
            if len(candidate_order) - start_idx < n_unseen - len(current_unseen):
                return None

            for candidate_idx in range(start_idx, len(candidate_order)):
                candidate_label = candidate_order[candidate_idx]
                if candidate_label in current_unseen:
                    continue
                if not is_valid_unseen(candidate_label, set(current_unseen)):
                    continue
                current_unseen.append(candidate_label)
                result = dfs(candidate_idx + 1, current_unseen)
                if result is not None:
                    return result
                current_unseen.pop()
            return None

        sampled = dfs(0, [])
        return sampled or []

    if len(candidates) <= n_unseen:
        unseen = []
        for label in candidates:
            if is_valid_unseen(label, set(unseen)):
                unseen.append(label)
    else:
        unseen = []
        for _ in range(64):
            sampled = sample_interval_candidates_once()
            if sampled and len(sampled) >= n_unseen:
                unseen = sampled[:n_unseen]
                break
        if len(unseen) < n_unseen:
            unseen = sample_with_backtracking()[:n_unseen]
        if len(unseen) < n_unseen:
            raise ValueError(
                "Unable to sample enough unseen labels under the current graph/semantic "
                f"constraints: selected={len(unseen)}, requested={n_unseen}, "
                f"candidate_count={len(candidates)}, "
                f"min_seen_neighbors={min_seen_neighbors}, "
                f"min_semantic_neighbors={min_semantic_neighbors}, "
                f"semantic_threshold={semantic_threshold}."
            )

    remaining = [label for label in candidates if label not in set(unseen)]

    for idx, label in enumerate(list(unseen)):
        current_unseen = set(unseen)
        semantic_seen = _semantic_seen_neighbors(
            label=label,
            nodes=nodes,
            label_to_index=label_to_index,
            similarity=similarity,
            excluded=current_unseen,
            threshold=semantic_threshold,
        )
        if seen_neighbor_count(label, current_unseen) >= min_seen_neighbors and len(semantic_seen) >= min_semantic_neighbors:
            continue

        current_unseen.remove(label)
        replacement_index = None
        for candidate_idx, candidate in enumerate(remaining):
            candidate_unseen = set(current_unseen)
            candidate_unseen.add(candidate)
            candidate_semantic_seen = _semantic_seen_neighbors(
                label=candidate,
                nodes=nodes,
                label_to_index=label_to_index,
                similarity=similarity,
                excluded=candidate_unseen,
                threshold=semantic_threshold,
            )
            if seen_neighbor_count(candidate, candidate_unseen) >= min_seen_neighbors and len(candidate_semantic_seen) >= min_semantic_neighbors:
                replacement_index = candidate_idx
                break
        if replacement_index is not None:
            unseen[idx] = remaining.pop(replacement_index)

    unseen = sorted(set(unseen), key=lambda label: (freq[label], label))
    if len(unseen) < n_unseen:
        raise ValueError(f"Only selected {len(unseen)} unseen labels, but requested n_unseen={n_unseen}.")
    seen = [label for label in nodes if label not in set(unseen)]
    seen_set = set(seen)
    unseen_set = set(unseen)

    unseen_info: list[dict[str, object]] = []
    for label in unseen:
        relation_seen_neighbors = {
            relation_name: sorted(neighbor for neighbor in rel_neighbors if neighbor in seen_set)
            for relation_name, rel_neighbors in relation_neighbors[label].items()
        }
        semantic_seen = _semantic_seen_neighbors(
            label=label,
            nodes=nodes,
            label_to_index=label_to_index,
            similarity=similarity,
            excluded=unseen_set,
            threshold=semantic_threshold,
        )
        unseen_info.append(
            {
                "label": label,
                "frequency": int(freq[label]),
                "kg_degree": int(degree[label]),
                "n_seen_neighbors": int(len({neighbor for values in relation_seen_neighbors.values() for neighbor in values})),
                "n_semantic_seen_neighbors": int(len(semantic_seen)),
                "seen_neighbors": relation_seen_neighbors,
                "nearest_seen": [[other_label, round(score, 4)] for other_label, score in semantic_seen[:5]],
                "clip_nearest_seen": [
                    {"label": other_label, "similarity": round(score, 4)} for other_label, score in semantic_seen[:5]
                ],
            }
        )
    return seen, unseen, unseen_info


def _resolve_output_targets(args: argparse.Namespace) -> list[tuple[int, Path]]:
    seeds = list(args.seeds) if args.seeds else [int(args.seed)]
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return [(seed, output_dir / f"split_seed{seed}.json") for seed in seeds]
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(seeds) != 1:
            raise ValueError("When using --output, provide exactly one seed. Use --output_dir for multiple seeds.")
        return [(seeds[0], output_path)]
    raise ValueError("Provide either --output or --output_dir.")


def save_split(payload: dict[str, object], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    output_targets = _resolve_output_targets(args)

    if args.legacy_selection:
        from utils.label_config_factory import load_label_config

        label_config_module = load_label_config(args.dataset)
        excluded_labels = sorted(getattr(label_config_module, "EXCLUDED_LABELS", set()))
        kg_definition = load_kg(args.kg_path)
        label_names = list(kg_definition["nodes"])
        anchors = load_text_anchors(args.text_anchors, label_names)
        vocabulary_provenance = {
            "path": str(Path(args.kg_path).resolve()),
            "source": "legacy_kg_definition",
            "label_names_sha256": _canonical_json_sha256(label_names),
        }
    else:
        excluded_labels = []
        if not args.label_names_file:
            raise ValueError(
                "Strict mode requires --label_names_file. Use --legacy-selection explicitly "
                "only to reproduce the old KG-conditioned protocol."
            )
        label_names, vocabulary_provenance = load_label_names(args.label_names_file)
        kg_definition = None
        anchors = None

    collection_entries: list[dict[str, object]] = []

    for seed, output_path in output_targets:
        if args.legacy_selection:
            seen, unseen, unseen_info = select_unseen_labels_legacy(
                kg_definition=kg_definition,
                text_anchors=anchors,
                n_unseen=args.n_unseen,
                seed=seed,
                min_seen_neighbors=args.min_seen_neighbors,
                min_semantic_neighbors=args.min_semantic_neighbors,
                semantic_threshold=args.semantic_threshold,
                protected_top_k=args.protected_top_k,
            )
        else:
            seen, unseen, unseen_info = select_unseen_labels(
                label_names=label_names,
                n_unseen=args.n_unseen,
                seed=seed,
            )

        print(f"\nSeed {seed}: seen={len(seen)}, unseen={len(unseen)}")
        print(f"Unseen labels for seed {seed}:")
        for item in unseen_info:
            if args.legacy_selection:
                nearest = ", ".join(
                    f"{neighbor['label']}({neighbor['similarity']:.2f})"
                    for neighbor in item["clip_nearest_seen"][:2]
                )
                print(
                    f"  {item['label']}: freq={item['frequency']}, degree={item['kg_degree']}, "
                    f"seen_neighbors={item['n_seen_neighbors']}, nearest={nearest}"
                )
            else:
                print(f"  {item['label']}")

        payload = {
            "schema_version": 2,
            "protocol_mode": "legacy_conditioned" if args.legacy_selection else "strict_outer_random_v1",
            "selection_policy": (
                "kg_frequency_semantic_conditioned"
                if args.legacy_selection
                else "uniform_without_replacement"
            ),
            "dataset": args.dataset,
            "seed": int(seed),
            "rng": "numpy.random.Generator(PCG64)",
            "n_labels": len(label_names),
            "n_seen": len(seen),
            "n_unseen": len(unseen),
            "excluded": excluded_labels,
            "label_names": label_names,
            "label_vocabulary": vocabulary_provenance,
            "selection_inputs": (
                ["kg_edges", "label_frequency", "text_anchor_similarity", "protected_top_k"]
                if args.legacy_selection
                else ["label_names", "seed", "n_unseen"]
            ),
            "seen_labels": seen,
            "unseen_labels": unseen,
            "unseen_info": unseen_info,
        }
        if args.legacy_selection:
            payload.update(
                {
                    "kg_path": str(Path(args.kg_path).resolve()),
                    "text_anchors": str(Path(args.text_anchors).resolve()),
                    "min_seen_neighbors": int(args.min_seen_neighbors),
                    "min_semantic_neighbors": int(args.min_semantic_neighbors),
                    "semantic_threshold": float(args.semantic_threshold),
                    "protected_top_k": int(args.protected_top_k),
                }
            )
        split_sha256 = save_split(payload, output_path)
        collection_entries.append(
            {
                "seed": int(seed),
                "path": str(output_path),
                "sha256": split_sha256,
                "seen_labels": seen,
                "unseen_labels": unseen,
            }
        )
        print(f"Saved: {output_path}")

    if args.output_dir:
        collection_path = Path(args.output_dir).resolve() / "outer_split_manifest.json"
        collection_payload = {
            "schema_version": 1,
            "protocol_mode": "legacy_conditioned" if args.legacy_selection else "strict_outer_random_v1",
            "dataset": args.dataset,
            "seeds": [int(item["seed"]) for item in collection_entries],
            "n_unseen": int(args.n_unseen),
            "label_vocabulary": vocabulary_provenance,
            "splits": collection_entries,
        }
        save_split(collection_payload, collection_path)
        print(f"Saved: {collection_path}")


if __name__ == "__main__":
    main()
