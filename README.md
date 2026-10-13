# Score-TEGAR

Code and protocol materials for *Score-Space Knowledge Graph Propagation for Generalized Multi-Label Zero-Shot Classification of Remote Sensing Images*.

## Included materials

- [Label splits, graphs and text anchors](artifacts/strict_v1/README.md) for MLRSNet, BigEarthNet and China-MAS: five outer splits per dataset, their inner splits, and the corresponding graphs. Text anchors are shared across splits within each dataset.
- `select_unseen.py`, `build_kg.py`, `build_semantic_affinity.py` and `build_text_anchors.py`: split generation and graph/anchor construction.
- `run_strict_protocol.py`, `train.py` and `evaluate.py`: inner-validation model selection, fixed-epoch outer training and evaluation, with the required model and data-loading code.
- Tests for label separation, graph scope and model-selection behavior.

The release contains no image data, per-image annotations, visual feature caches, trained model weights, predictions, training logs or experimental result files. Graphs retain their label order and edges, while local paths and annotation-frequency summaries are omitted. Text-anchor tensors and label-split memberships are unchanged.

## Check the supplied materials

Use Python 3.10 or later and install `requirements.txt`:

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_released_artifacts.py
```

This checks the released file hashes, inner/outer label separation, empirical-edge endpoints and text-only connections to unseen labels. It does not require the image datasets.

## Run the protocol with local data

Set the dataset, feature-cache and frozen GeoRSCLIP checkpoint paths in `configs/strict_a.yaml`. Use `build_feature_cache.py` and `scripts/audit_feature_cache_provenance.py` to prepare and verify local train/validation/test feature caches.

```bash
python run_strict_protocol.py --config configs/strict_a.yaml --phase prepare
python run_strict_protocol.py --config configs/strict_a.yaml --phase audit
python run_strict_protocol.py --config configs/strict_a.yaml --phase full --methods score_tegar
```

Empirical graph construction uses only training images and the current seen-label columns. Inner validation draws surrogate-unseen labels from the outer-seen set. The selected epoch count, residual scale and decision threshold are fixed for outer evaluation. Generated files stay in the configured local output directories.
