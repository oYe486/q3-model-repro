"""Read-only attachment-4 schema/pairing check; writes a metadata report only.

This deliberately does not run sentiment inference, fit thresholds, select a
model, or treat feature indices as original media timestamps.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import BertTokenizerFast

from .attachment4_data import load_aligned_sample, load_feature_pickle


EXPECTED_KEYS = {
    '对齐版本': {'id', 'raw_text', 'text', 'text_bert', 'audio', 'vision'},
    '未对齐版本': {'id', 'raw_text', 'text', 'text_bert', 'audio', 'vision',
               'audio_lengths', 'vision_lengths'},
}
EXPECTED_SHAPES = {
    '对齐版本': {'text': (50, 768), 'text_bert': (3, 50),
             'audio': (50, 74), 'vision': (50, 35)},
    '未对齐版本': {'text': (50, 768), 'text_bert': (3, 50),
              'audio': (500, 74), 'vision': (500, 35)},
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inspect(root):
    tokenizer = BertTokenizerFast.from_pretrained(
        str(Path(__file__).resolve().parents[1] / 'pretrained/bert-base-uncased'))
    by_version, raw_text_by_id, token_ids_by_id, videos_by_id = {}, {}, {}, {}
    ids_by_version, warnings = {}, []
    for version in ('对齐版本', '未对齐版本'):
        folder = root / version
        feature_files = sorted(folder.glob('*.pkl'))
        video_files = sorted((folder / 'videos').glob('*.mp4'))
        feature_stems = {path.stem for path in feature_files}
        video_stems = {path.stem for path in video_files}
        if not feature_files or feature_stems != video_stems:
            raise ValueError(f'{version}: feature/video IDs do not pair exactly.')
        records = []
        dtypes = {}
        for feature_path in feature_files:
            data = load_feature_pickle(feature_path)
            if not isinstance(data, dict) or set(data) != EXPECTED_KEYS[version]:
                raise ValueError(f'{feature_path}: unexpected field set {set(data)}.')
            for name, shape in EXPECTED_SHAPES[version].items():
                if np.shape(data[name]) != shape:
                    raise ValueError(f'{feature_path}: {name} shape is not {shape}.')
                values = np.asarray(data[name])
                if not np.isfinite(values).all():
                    raise ValueError(f'{feature_path}: {name} contains NaN/Inf.')
                dtypes.setdefault(name, set()).add(str(values.dtype))
            sample_id = str(data['id'])
            if sample_id != feature_path.stem:
                raise ValueError(f'{feature_path}: ID and filename disagree.')
            raw_text = str(data['raw_text'])
            encoded = tokenizer(raw_text, max_length=50, truncation=True,
                                padding='max_length')
            text_bert = np.asarray(data['text_bert'])
            expected = np.stack((encoded['input_ids'], encoded['attention_mask'],
                                 encoded['token_type_ids']))
            token_match = bool(np.array_equal(text_bert, expected))
            if not token_match:
                raise ValueError(f'{feature_path}: raw_text does not match text_bert.')
            video_path = folder / 'videos' / f'{sample_id}.mp4'
            if video_path.stat().st_size <= 0:
                raise ValueError(f'{video_path}: empty video.')
            raw_text_by_id.setdefault(sample_id, raw_text)
            token_ids_by_id.setdefault(sample_id, text_bert[0].copy())
            if (raw_text_by_id[sample_id] != raw_text or not
                    np.array_equal(token_ids_by_id[sample_id], text_bert[0])):
                raise ValueError(f'{sample_id}: text differs across versions.')
            video_hash = digest(video_path)
            videos_by_id.setdefault(sample_id, video_hash)
            if videos_by_id[sample_id] != video_hash:
                raise ValueError(f'{sample_id}: videos differ across versions.')
            record = {
                'id': sample_id,
                'feature_file': str(feature_path.resolve()),
                'video_file': str(video_path.resolve()),
                'video_bytes': video_path.stat().st_size,
                'text_valid_tokens': int(np.asarray(data['text_bert'])[1].sum()),
                'raw_text_bert_tokens_without_truncation': len(
                    tokenizer(raw_text)['input_ids']),
                'raw_text_tokenization_matches': token_match,
            }
            if version == '对齐版本':
                sample = load_aligned_sample(feature_path)
                record['audio_observed_aligned_positions'] = int(
                    sample['audio_mask'].sum())
                record['vision_observed_aligned_positions'] = int(
                    sample['vision_mask'].sum())
                for name in ('audio', 'vision'):
                    if record[f'{name}_observed_aligned_positions'] == 0:
                        warnings.append({
                            'id': sample_id, 'version': version,
                            'kind': f'{name}_unavailable_after_training_mask',
                            'detail': 'No nonzero observed aligned feature row.'})
                if record['raw_text_bert_tokens_without_truncation'] > 50:
                    warnings.append({
                        'id': sample_id, 'version': version,
                        'kind': 'raw_text_truncated_by_model',
                        'raw_tokens': record['raw_text_bert_tokens_without_truncation'],
                        'modeled_tokens': 50,
                        'detail': 'Evidence may only cite the modeled prefix.'})
            else:
                for name in ('audio', 'vision'):
                    length = int(data[f'{name}_lengths'])
                    if not 0 < length <= 500:
                        raise ValueError(f'{feature_path}: invalid {name} length.')
                    record[f'{name}_valid_length'] = length
                    # Report only: the supplied length, not index/time scaling,
                    # is the source of valid independent sequence positions.
                    record[f'{name}_nonzero_rows_beyond_length'] = int(
                        np.any(np.asarray(data[name])[length:] != 0, axis=1).sum())
                    if record[f'{name}_nonzero_rows_beyond_length']:
                        warnings.append({
                            'id': sample_id, 'version': version,
                            'kind': f'{name}_length_conflicts_with_nonzero_rows',
                            'reported_length': length,
                            'nonzero_rows_beyond_length':
                                record[f'{name}_nonzero_rows_beyond_length']})
            records.append(record)
        ids_by_version[version] = feature_stems
        by_version[version] = {
            'feature_count': len(feature_files),
            'video_count': len(video_files),
            'field_names': sorted(EXPECTED_KEYS[version]),
            'feature_shapes': {name: list(shape) for name, shape in
                               EXPECTED_SHAPES[version].items()},
            'feature_dtypes': {name: sorted(values) for name, values in
                               dtypes.items()},
            'samples': records,
        }
    if ids_by_version['对齐版本'] != ids_by_version['未对齐版本']:
        raise AssertionError('Cross-version sample IDs do not match.')
    return {
        'inspection_only': True,
        'n_paired_samples': len(raw_text_by_id),
        'versions': by_version,
        'compatible_model_input': '对齐版本 only for the current aligned checkpoint',
        'labels_present': False,
        'explicit_feature_to_media_timestamps_present': False,
        'media_available': True,
        'warnings': warnings,
        'localization_remaining': [
            'align raw transcript to MP4 audio for word-level times',
            'map audio/vision feature windows to verified media intervals',
            'select and verify actual visual frames from MP4',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=
                        Path(__file__).resolve().parents[2] /
                        'datasets/附件4-可解释专项视频样本与特征文件')
    parser.add_argument('--output', type=Path, default=
                        Path(__file__).resolve().parents[1] /
                        'outputs/attachment4_structure/manifest.json')
    args = parser.parse_args()
    manifest = inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        'n_paired_samples': manifest['n_paired_samples'],
        'versions': {name: {key: value for key, value in details.items()
                            if key != 'samples'}
                     for name, details in manifest['versions'].items()},
        'explicit_feature_to_media_timestamps_present': False,
        'manifest': str(args.output),
    }, ensure_ascii=False, indent=2))
