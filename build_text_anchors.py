from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.kg_builder import KGDefinition





def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP text anchors for dataset labels.")
    parser.add_argument("--dataset", default="mlrsnet", choices=["mlrsnet", "bigearthnet", "chinamas"])
    parser.add_argument("--kg_path", default="./data/kg_definition.json")
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default=None)
    parser.add_argument("--clip_default_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--output", default="./data/text_anchors.pt")
    parser.add_argument("--manifest_output", default=None)
    parser.add_argument(
        "--legacy-cooccur-prompts",
        action="store_true",
        help="Explicitly enable the legacy KG-co-occurrence-conditioned prompt construction.",
    )
    if argv is not None:
        return parser.parse_args(argv)
    return parser.parse_args()


def load_label_names(kg_path: str | Path) -> list[str]:
    return list(KGDefinition(kg_path).nodes)


def build_label_anchors(
    kg_definition: KGDefinition,
    encoder,
    label_config_module,
    legacy_cooccur_prompts: bool = False,
) -> torch.Tensor:
    templates = list(getattr(label_config_module, "TEMPLATES"))
    cooccur_templates = list(getattr(label_config_module, "COOCCUR_TEMPLATES", templates))
    label_name_fix = dict(getattr(label_config_module, "LABEL_NAME_FIX", {}))

    def get_display_names(label: str) -> list[str]:
        aliases = label_name_fix.get(label, [label])
        return [str(alias).replace("_", " ") for alias in aliases]

    def build_label_prompts(label: str) -> list[str]:
        prompts: list[str] = []
        for display_name in get_display_names(label):
            prompts.extend(template.format(display_name) for template in templates)
        return prompts

    def build_cooccur_prompts(label: str, cooccur_labels: list[str], top_k: int = 3) -> list[str]:
        names = get_display_names(label)
        context_names: list[str] = []
        for neighbor_label in cooccur_labels[:top_k]:
            context_names.extend(get_display_names(neighbor_label))
        if not context_names:
            return build_label_prompts(label)
        context = ", ".join(context_names[:top_k])
        prompts: list[str] = []
        for display_name in names:
            prompts.extend(template.format(display_name, context) for template in cooccur_templates)
        prompts.extend(build_label_prompts(label))
        return prompts

    anchors: list[torch.Tensor] = []
    cooccur_lookup: dict[str, list[str]] = {label: [] for label in kg_definition.nodes}
    if legacy_cooccur_prompts:
        cooccur_edges = kg_definition.edges.get("often_cooccur", [])
        for src, dst in cooccur_edges:
            if dst not in cooccur_lookup[src]:
                cooccur_lookup[src].append(dst)
            if src not in cooccur_lookup[dst]:
                cooccur_lookup[dst].append(src)

    for label in kg_definition.nodes:
        if legacy_cooccur_prompts:
            neighbor_labels = cooccur_lookup.get(label, [])
            texts = build_cooccur_prompts(label, neighbor_labels, top_k=3)
        else:
            texts = build_label_prompts(label)
        template_features = encoder.encode_text(texts)
        anchor = F.normalize(template_features.mean(dim=0), dim=0)
        anchors.append(anchor)
    return torch.stack(anchors, dim=0)


def compute_anchor_sha256(label_names: list[str], anchors: torch.Tensor) -> str:
    tensor = anchors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(label_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def print_similarity_summary(label_names: list[str], anchors: torch.Tensor) -> None:
    similarity = anchors @ anchors.T
    for idx, label in enumerate(label_names):
        scores = similarity[idx].clone()
        scores[idx] = -1.0
        top_values, top_indices = torch.topk(scores, k=min(3, len(label_names) - 1))
        formatted = ", ".join(
            f"{label_names[top_idx]}({float(top_val):.2f})"
            for top_idx, top_val in zip(top_indices.tolist(), top_values.tolist())
        )
        print(f"Label '{label}': anchor norm={float(anchors[idx].norm()):.4f}, top-3 similar: [{formatted}]")


def main() -> None:
    args = parse_args()

    from models.clip_backbone import CLIPTextEncoder
    from utils.label_config_factory import load_label_config

    label_names = load_label_names(args.kg_path)
    label_config_module = load_label_config(args.dataset)
    templates = list(getattr(label_config_module, "TEMPLATES"))
    cooccur_templates = list(getattr(label_config_module, "COOCCUR_TEMPLATES", templates))
    print(f"Loaded KG: {len(label_names)} labels")
    if args.legacy_cooccur_prompts:
        print(
            "Encoding LEGACY text anchors with "
            f"{len(templates)} base templates + {len(cooccur_templates)} co-occurrence templates..."
        )
    else:
        print(f"Encoding strict basic-only text anchors with {len(templates)} base templates...")

    encoder = CLIPTextEncoder(
        model_name=args.clip_model,
        pretrained_path=args.clip_pretrained,
        default_pretrained=args.clip_default_pretrained,
    )
    kg_definition = KGDefinition(args.kg_path)
    anchors = build_label_anchors(
        kg_definition,
        encoder,
        label_config_module,
        legacy_cooccur_prompts=args.legacy_cooccur_prompts,
    )
    print_similarity_summary(label_names, anchors)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_sha256 = compute_anchor_sha256(label_names, anchors)
    anchor_mode = "legacy_cooccurrence_conditioned" if args.legacy_cooccur_prompts else "strict_basic_only_v1"
    templates_used = (
        list(dict.fromkeys(templates + cooccur_templates))
        if args.legacy_cooccur_prompts
        else list(templates)
    )
    manifest = {
        "schema_version": 1,
        "protocol_mode": anchor_mode,
        "dataset": args.dataset,
        "kg_path": str(Path(args.kg_path).resolve()),
        "label_names": label_names,
        "label_names_sha256": hashlib.sha256(
            json.dumps(label_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "anchor_sha256": anchor_sha256,
        "anchor_shape": list(anchors.shape),
        "base_templates_used": templates,
        "cooccur_templates_used": cooccur_templates if args.legacy_cooccur_prompts else [],
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "clip_default_pretrained": args.clip_default_pretrained,
    }
    torch.save(
        {
            "anchors": anchors.cpu(),
            "label_names": label_names,
            "templates_used": templates_used,
            "base_templates_used": templates,
            "cooccur_templates_used": cooccur_templates if args.legacy_cooccur_prompts else [],
            "dataset": args.dataset,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "clip_default_pretrained": args.clip_default_pretrained,
            "protocol_mode": anchor_mode,
            "anchor_sha256": anchor_sha256,
            "manifest": manifest,
        },
        output_path,
    )
    manifest_path = (
        Path(args.manifest_output).resolve()
        if args.manifest_output
        else output_path.with_name(f"{output_path.stem}_manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}, shape={tuple(anchors.shape)}, sha256={anchor_sha256}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
