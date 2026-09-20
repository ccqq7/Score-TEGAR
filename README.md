# Score-TEGAR

Main model implementation for remote-sensing zero-shot multi-label classification, synchronized with the revised local model on 2026-09-20.

This repository contains the model and its supporting Python modules. Training, evaluation, dataset preparation, experiment orchestration, data, checkpoints, and results are outside this release's scope.

## Current Model

`ProbabilitySpaceRefiner` refines frozen CLIP/GeoRSCLIP label scores using a typed label graph:

```text
Frozen visual features + label text anchors
-> initial label scores
-> score + uncertainty + text-anchor node encoding
-> typed graph propagation and scene-conditioned relation experts
-> uncertainty-gated residual score correction
-> temperature-scaled multi-label logits
```

The revised graph supports four relations: `often_cooccur`, `statistical_exclusion`, `hierarchical`, and `semantic_affinity`. Supply a KG containing `semantic_affinity` edges to use the four-relation model; a KG without that relation retains three relation branches.

With the default `exclusion_mode="negative"` and `strict_exclusion_score=True`, statistical exclusion has a separate, non-positive score contribution. Its arbitrary-signed hidden message and expert heads do not contribute to the correction. This constraint applies to the exclusion branch; the complete residual can still increase or decrease a score through the other branches.

The model also exposes relation ablations, learned/fixed/shuffled gates, and a parameter-matched homogeneous graph control (`graph_type="homo_pm"`).

## Setup

Use Python 3.10+ and install the model dependencies:

```bash
pip install -r requirements.txt
```

## Core Entry Point

```python
from models.logit_refiner import ProbabilitySpaceRefiner

model = ProbabilitySpaceRefiner(
    num_labels=num_labels,
    kg_path="data/kg_definition.json",
    text_anchors_path="data/text_anchors.pt",
    clip_model_name="ViT-B-32",
    clip_pretrained_path="pretrained/georsclip/model.pt",
    graph_type="tegar",
    hidden_dim=128,
    num_layers=2,
    exclusion_mode="negative",
    strict_exclusion_score=True,
)
model.eval()
```

For precomputed visual features, construct the model with `use_cached_features=True` and call `model.forward_from_features(features)`. In this mode, a visual checkpoint is not loaded. Features must have shape `[batch_size, anchor_dim]` and come from the backbone paired with the text anchors; move the model and features to the same device.

Pass `return_aux=True` to obtain diagnostics, including relation gates and the explicit/applied exclusion score contributions.

## Required Artifacts

- `kg_definition.json`: a label graph with `nodes`, `node_count`, `label_to_index`, `label_frequency`, and `edges`. Each relation's edge list contains `[source_label, target_label]` pairs. Co-occurrence, exclusion, and semantic-affinity edges are symmetrized; hierarchy edges retain their direction.
- `text_anchors.pt`: a PyTorch dictionary containing a two-dimensional `anchors` tensor and `label_names`. Anchor rows and labels must match the KG node order after label filtering, and their count must equal `num_labels`. The inherited label helper excludes `stadium`.
- A CLIP/GeoRSCLIP checkpoint for raw-image input, or an `open_clip` pretrained tag supplied as `clip_default_pretrained`. Cached-feature input does not require loading this checkpoint.

These generated artifacts are not included. The repository loads supplied graphs and anchors; it does not provide the experimental split-generation or preprocessing pipeline.

## Files

- `models/logit_refiner.py`: score refinement, relation experts, uncertainty gating, and the strict exclusion score path.
- `models/tegar.py`: typed graph layers, four-relation support, gate controls, and homogeneous graph controls.
- `models/classifier.py`: model wrappers, the model factory, and `load_text_anchors()`.
- `models/controlled_baselines.py`, `models/baselines.py`: model dependencies referenced by the updated factory.
- `models/clip_backbone.py`: frozen CLIP/GeoRSCLIP backbone and input transforms.
- `models/backbone.py`: ResNet dependency used by classifier wrappers.
- `utils/kg_builder.py`: graph loading and adjacency construction.
- `utils/label_config.py`: label helpers and KG filtering.

## Compatibility

The public implementation before this synchronization used three relations and a different exclusion mechanism. The revised model adds parameters and changes gate dimensions when semantic-affinity edges are present. Earlier trained Score-TEGAR checkpoints are not directly interchangeable with the revised model; use weights produced with the matching architecture and graph. This does not change the external CLIP/GeoRSCLIP backbone checkpoint format.
