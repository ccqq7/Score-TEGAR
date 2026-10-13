import hashlib
import json
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / 'artifacts' / 'strict_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_released_splits_graphs_and_anchors():
    manifest = read(ROOT / 'manifest.json')
    assert len(manifest['files']) == 66
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.is_file() and path.stat().st_size == item['bytes']
        assert digest(path) == item['sha256']
    for dataset in ('mlrsnet', 'bigearthnet', 'chinamas'):
        folder = ROOT / dataset
        anchor_path = folder / f'text_anchors_{dataset}_basic.pt'
        payload = torch.load(anchor_path, map_location='cpu', weights_only=True)
        labels = payload['label_names']
        assert len(set(labels)) == len(labels)
        assert payload['anchors'].shape == (len(labels), 512)
        assert torch.isfinite(payload['anchors']).all()
        assert not payload['cooccur_templates_used']
        for seed in manifest['outer_seeds']:
            outer_path = folder / 'outer_splits' / f'split_seed{seed}.json'
            inner_path = folder / f'outer_{seed}' / 'inner_split.json'
            outer, inner = read(outer_path), read(inner_path)
            assert set(outer['seen_labels']).isdisjoint(outer['unseen_labels'])
            assert set(outer['seen_labels']) | set(outer['unseen_labels']) == set(labels)
            assert set(inner['outer_unseen_labels']) == set(outer['unseen_labels'])
            assert set(inner['selection_unseen_labels']) <= set(outer['seen_labels'])
            assert set(inner['seen_labels']).isdisjoint(inner['unseen_labels'])
            assert set(inner['selection_seen_labels']) | set(inner['selection_unseen_labels']) == set(outer['seen_labels'])
            for stage, split_path, split in [('outer', outer_path, outer), ('inner', inner_path, inner)]:
                graph = read(folder / f'outer_{seed}' / f'{stage}_graph.json')
                assert graph['nodes'] == labels
                seen = set(split['seen_labels'])
                for relation in ('often_cooccur', 'statistical_exclusion', 'hierarchical'):
                    assert all(len(edge) == 2 and set(edge) <= seen for edge in graph['edges'][relation])
                assert set(graph['edge_provenance']['allowed_seen_labels']) == seen
                assert graph['edge_provenance']['allowed_seen_labels_provenance']['file_sha256'] == digest(split_path)
                assert graph['edge_provenance']['source']['image_split_manifest']['declared_split'] == 'train'
                semantic = graph['relation_provenance']['semantic_affinity']
                assert semantic['annotation_independent'] is True
                assert semantic['text_anchors_sha256'] == digest(anchor_path)
                assert semantic['outer_split_sha256'] == digest(split_path)
                for label in split['unseen_labels']:
                    neighbors = {v if u == label else u for u, v in graph['edges']['semantic_affinity'] if label in (u, v)}
                    assert len(neighbors) == 3 and neighbors <= seen


def test_release_excludes_experiment_data_and_private_metadata():
    forbidden_keys = {
        'path', 'source_files', 'kg_path', 'clip_pretrained', 'file_size', 'file_mtime_ns',
        'train_sample_count', 'sample_count', 'total_images', 'train_sample_ids_sha256',
        'seen_label_matrix_sha256', 'label_frequency', 'label_frequency_scope',
        'metrics', 'predictions', 'model_state_dict', 'optimizer_state_dict',
        'image_paths', 'image_names', 'features', 'annotations',
    }

    def check_metadata(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for item in value.values():
                check_metadata(item)
        elif isinstance(value, list):
            for item in value:
                check_metadata(item)
        elif isinstance(value, str):
            assert not re.search(r'[A-Za-z]:[\\/]|/(?:home|mnt|Users)/', value)

    manifest = read(ROOT / 'manifest.json')
    expected = {item['path'] for item in manifest['files']} | {'manifest.json', 'README.md'}
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob('*') if path.is_file()}
    assert actual == expected
    for item in manifest['files']:
        path = ROOT / item['path']
        if path.suffix == '.pt':
            assert path.name == f'text_anchors_{path.parent.name}_basic.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert set(key for key, value in payload.items() if isinstance(value, torch.Tensor)) == {'anchors'}
        else:
            assert path.suffix == '.json'
            payload = read(path)
        check_metadata(payload)
