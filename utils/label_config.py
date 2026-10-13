from __future__ import annotations

from copy import deepcopy


EXCLUDED_LABELS: set[str] = {"stadium"}


LABEL_NAME_FIX: dict[str, list[str]] = {
    "habor": ["harbor"],
    "bare soil": ["bare soil", "barren land"],
    "gully": ["gully", "ravine"],
    "chaparral": ["chaparral", "shrubland"],
    "track": ["track", "dirt trail"],
    "terrace": ["terrace", "terraced farmland"],
}


TEMPLATES: list[str] = [
    "a satellite image of {}",
    "an aerial photograph of {}",
    "a remote sensing image of {}",
    "a satellite image with {} in it",
    "an aerial photograph containing {}",
    "{} visible in a remote sensing scene",
]


# DualPrompt templates: inject co-occurring label context into prompts.
# Reference: "Unlocking the Power of Co-Occurrence in CLIP" (2025)
COOCCUR_TEMPLATES: list[str] = [
    "a satellite image of {} with {}",
    "a remote sensing image showing {} near {}",
    "an aerial view of {} alongside {}",
    "a satellite image containing {} and {}",
]


def get_display_names(label: str) -> list[str]:
    aliases = LABEL_NAME_FIX.get(label, [label])
    return [alias.replace("_", " ") for alias in aliases]


def build_label_prompts(label: str, templates: list[str] | None = None) -> list[str]:
    templates = templates or TEMPLATES
    prompts: list[str] = []
    for name in get_display_names(label):
        prompts.extend(template.format(name) for template in templates)
    return prompts


def build_cooccur_prompts(
    label: str,
    cooccur_labels: list[str],
    top_k: int = 3,
    templates: list[str] | None = None,
) -> list[str]:
    """Build prompts that include co-occurring labels for richer semantics.

    If no co-occurring labels are available, falls back to standard prompts.
    """
    templates = templates or COOCCUR_TEMPLATES
    prompts: list[str] = []
    names = get_display_names(label)
    cooccur_names = []
    for co_label in cooccur_labels[:top_k]:
        cooccur_names.extend(get_display_names(co_label))

    if not cooccur_names:
        return build_label_prompts(label)

    context = ", ".join(cooccur_names[:top_k])
    for name in names:
        prompts.extend(template.format(name, context) for template in templates)
    # Also include standard prompts to maintain baseline representation
    prompts.extend(build_label_prompts(label))
    return prompts


def get_active_labels(all_labels: list[str]) -> list[str]:
    return [label for label in all_labels if label not in EXCLUDED_LABELS]


def filter_edges(edges: dict[str, list[list[str]]]) -> dict[str, list[list[str]]]:
    filtered: dict[str, list[list[str]]] = {}
    for relation_name, relation_edges in edges.items():
        filtered[relation_name] = [
            [src, dst]
            for src, dst in relation_edges
            if src not in EXCLUDED_LABELS and dst not in EXCLUDED_LABELS
        ]
    return filtered


def filter_kg_payload(payload: dict) -> dict:
    filtered = deepcopy(payload)
    active_nodes = get_active_labels(list(payload["nodes"]))
    filtered["nodes"] = active_nodes
    filtered["node_count"] = len(active_nodes)
    filtered["edges"] = filter_edges(dict(payload["edges"]))
    filtered["label_to_index"] = {label: idx for idx, label in enumerate(active_nodes)}
    filtered["label_frequency"] = {
        label: int(payload["label_frequency"][label])
        for label in active_nodes
        if label in payload.get("label_frequency", {})
    }
    if "merged_labels" in payload:
        filtered["merged_labels"] = [
            item
            for item in payload["merged_labels"]
            if item.get("target") not in EXCLUDED_LABELS and item.get("source") not in EXCLUDED_LABELS
        ]
    return filtered
