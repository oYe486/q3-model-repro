"""Local aligned-feature occlusion and decision-margin utilities."""

import numpy as np


NAMES = ('text', 'audio', 'vision')


def class_margin(score, predicted_class, lower, upper):
    if predicted_class == 0:
        return lower - score
    if predicted_class == 2:
        return score - upper
    return min(score - lower, upper - score)


def occlude(base, modality, start, end):
    variant = {name: value.clone() for name, value in base.items()}
    if modality == 'text':
        variant['text'][:, :, start:end] = 0
        variant['text_mask'][:, start:end] = False
    else:
        variant[modality][:, start:end] = 0
        variant[modality + '_mask'][:, start:end] = False
    return variant


def candidate_windows(masks, window_size):
    """Sweep non-overlapping windows over observed aligned positions."""
    windows = {}
    for name in NAMES:
        valid = masks[name][0].cpu().numpy().copy()
        if name == 'text':
            positions = np.flatnonzero(valid)
            valid[positions[0]] = False
            valid[positions[-1]] = False
        observed = np.flatnonzero(valid)
        windows[name] = ([(start, min(start + window_size, int(observed[-1]) + 1))
                          for start in range(int(observed[0]), int(observed[-1]) + 1,
                                             window_size)
                          if valid[start:start + window_size].any()]
                         if len(observed) else [])
    return windows
