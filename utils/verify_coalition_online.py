"""Verify online, label-free explanation matches validation records."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.model.DLF import DLF


def main():
    root = Path(__file__).resolve().parents[1]
    config = get_config_regression('DLF', 'mosei')
    config.feature_T = config.feature_A = config.feature_V = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MMDataset(config, mode='valid')
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=0)
    model = DLF.from_final_checkpoint(root=root, device=device)
    indices, scores, mains, supports = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            result = model(
                batch['text'].to(device), batch['audio'].to(device),
                batch['vision'].to(device),
                text_mask=batch['text_mask'].to(device),
                audio_mask=batch['audio_mask'].to(device),
                vision_mask=batch['vision_mask'].to(device),
                return_explanation=True)
            indices.extend(batch['index'].tolist())
            scores.extend(result['output_logit'].reshape(-1).cpu().tolist())
            mains.extend(result['main_modality'].cpu().tolist())
            supports.extend(result['shapley_decision_support'].cpu().tolist())
    saved = pd.read_csv(root / 'outputs/validation/valid_predictions.csv')
    saved = saved.set_index('sample_index').loc[indices]
    names = np.array(['text', 'audio', 'vision'])
    if not np.array_equal(names[np.asarray(mains)],
                          saved.main_modality.to_numpy()):
        raise AssertionError('Online main differs from validation record.')
    score_error = np.max(np.abs(np.asarray(scores) -
                                saved.predicted_score.to_numpy()))
    support_error = np.max(np.abs(
        np.asarray(supports) -
        saved[[f'shapley_support_{name}' for name in names]].to_numpy()))
    if max(score_error, support_error) > 1e-5:
        raise AssertionError('Online inference differs from validation record.')
    print(f'{len(indices)} online samples; max score error {score_error:.9g}; '
          f'max support error {support_error:.9g}')


if __name__ == '__main__':
    main()
