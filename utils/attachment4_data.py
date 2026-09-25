"""Attachment-4 input adapter; no labels, inference, or model selection.

The existing checkpoint was trained on aligned_50.pkl. This adapter therefore
accepts only attachment-4 aligned samples for model input. The unaligned files
are inspected separately, not silently converted or mixed with aligned data.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def load_feature_pickle(path):
    """Read the provided NumPy pickle with the project's NumPy 1.x runtime.

    Attachment 4 contains references to NumPy 2's ``numpy._core`` namespace.
    The project environment currently uses NumPy 1.x. Retry solely for that
    known namespace mismatch; do not change array contents or install NumPy 2
    into the trained model environment. As with all pickle files, only load
    files from the trusted supplied dataset.
    """
    path = Path(path)
    try:
        with path.open('rb') as stream:
            return pickle.load(stream)
    except ModuleNotFoundError as error:
        if not (error.name or '').startswith('numpy._core'):
            raise
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    with path.open('rb') as stream:
        return pickle.load(stream)


def load_aligned_sample(path):
    """Return one label-free sample matching the aligned training interface."""
    path = Path(path)
    data = load_feature_pickle(path)
    required = {'id', 'raw_text', 'text_bert', 'audio', 'vision'}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f'{path} is not an attachment-4 sample dictionary.')
    if any(key in data for key in ('labels', 'regression_labels',
                                    'classification_labels')):
        raise ValueError('Attachment-4 samples must remain label-free.')
    if str(data['id']) != path.stem:
        raise ValueError(f'{path}: sample ID and filename disagree.')
    text = np.asarray(data['text_bert'])
    audio = np.asarray(data['audio'], dtype=np.float32).copy()
    vision = np.asarray(data['vision'], dtype=np.float32).copy()
    if text.shape != (3, 50) or audio.shape != (50, 74) or vision.shape != (50, 35):
        raise ValueError(f'{path} does not match aligned_50 feature shapes.')
    if not np.isfinite(text).all() or not np.isfinite(vision).all():
        raise ValueError(f'{path} has non-finite text or visual features.')
    if np.isnan(audio).any() or np.isposinf(audio).any():
        raise ValueError(f'{path} has unsupported non-finite audio features.')
    audio[np.isneginf(audio)] = 0
    text_mask = text[1].astype(bool)
    text_length = int(text_mask.sum())
    if text_length < 2 or not text_mask[:text_length].all() or text_mask[text_length:].any():
        raise ValueError(f'{path} has an invalid BERT attention mask.')
    word_mask = text_mask.copy()
    word_mask[0] = False
    word_mask[text_length - 1] = False
    audio_mask = word_mask & np.any(np.isfinite(audio) & (audio != 0), axis=1)
    vision_mask = word_mask & np.any(np.isfinite(vision) & (vision != 0), axis=1)
    video_path = path.parent / 'videos' / f'{path.stem}.mp4'
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    return {
        'id': str(data['id']),
        'raw_text': str(data['raw_text']),
        'text': torch.from_numpy(text.astype(np.float32)),
        'audio': torch.from_numpy(audio),
        'vision': torch.from_numpy(vision),
        'text_mask': torch.from_numpy(text_mask.copy()),
        'audio_mask': torch.from_numpy(audio_mask.copy()),
        'vision_mask': torch.from_numpy(vision_mask.copy()),
        'feature_file': str(path.resolve()),
        'video_file': str(video_path.resolve()),
    }
