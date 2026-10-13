# Protocol materials

Each dataset directory contains:

- `outer_splits/split_seed*.json`: five outer label splits, with seeds 42, 123, 456, 789 and 1024.
- `outer_*/inner_split.json`: the corresponding inner seen and surrogate-unseen labels.
- `outer_*/inner_graph.json` and `outer_*/outer_graph.json`: graph topology and label-scope provenance.
- `text_anchors_*_basic.pt` and its JSON manifest: frozen label-name text anchors shared across splits.

`manifest.json` lists the SHA-256 hashes of these 66 files. The graph edges, label splits and anchor tensors are retained; local paths, sample counts and annotation-frequency summaries are omitted. No experimental results or image-level records are included.

From the repository root, run `python -m pytest -q tests/test_released_artifacts.py` to check the materials. The graph-construction scripts accept locally supplied training data for reconstruction.
