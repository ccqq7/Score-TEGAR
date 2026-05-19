<<<<<<< HEAD
# Score-TERAG
=======
# TEGAR Main Model Code

This folder contains only the main model-construction code for the final method. It excludes training scripts, evaluation scripts, dataset loaders, experiment orchestration, result analysis, logs, checkpoints, cached features, and paper-generation utilities.

## Core Entry Point

The main model is:

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
)
```

`ProbabilitySpaceRefiner` implements the final score-space refinement pipeline:

```text
CLIP/GeoRSCLIP visual feature
-> initial label scores
-> score + uncertainty + text-anchor node encoding
-> TEGAR typed graph propagation
-> residual score correction
-> refined multi-label logits
```

## Files

- `models/logit_refiner.py`: final model entry point. Builds the score-space node features, calls TEGAR, and predicts residual logit corrections.
- `models/tegar.py`: typed-edge graph propagation module with co-occurrence, statistical-exclusion, and hierarchy relations.
- `models/classifier.py`: model wrappers and `load_text_anchors()`, which is used by `ProbabilitySpaceRefiner`.
- `models/clip_backbone.py`: frozen CLIP/GeoRSCLIP visual backbone wrapper.
- `models/backbone.py`: legacy ResNet dependency needed by classifier wrappers.
- `utils/kg_builder.py`: `KGDefinition` loader and typed adjacency construction.
- `utils/label_config.py`: helper functions for active-label and KG filtering.

## Required Artifacts

To instantiate the final model, provide:

- `kg_definition.json`: typed label knowledge graph.
- `text_anchors.pt`: label text-anchor tensor and label names.
- CLIP/GeoRSCLIP checkpoint, or an `open_clip` pretrained model name.

These artifacts are not included because they are generated data/model files rather than source code.

## Dependencies

Install the minimal Python dependencies:

```bash
pip install -r requirements.txt
```

The code expects PyTorch and `open_clip_torch` for CLIP/GeoRSCLIP loading.

>>>>>>> b3b1843 (提交文件)
