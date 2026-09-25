"""Check latent coalitions equal independently masked-input predictions."""

from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.model.DLF import DLF


def main(limit=16, tolerance=1e-5):
    root = Path(__file__).resolve().parents[1]
    config = get_config_regression('DLF', 'mosei')
    config.adaptive_main = True
    config.coalition_aware = True
    config.gate_temperature = 1.0
    config.feature_T = config.feature_A = config.feature_V = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DLF(config).to(device)
    model.load_state_dict(torch.load(
        root / 'checkpoints/final/model.pth',
        map_location=device), strict=True)
    model.eval()
    dataset = MMDataset(config, mode='valid')
    max_difference = 0.0
    comparisons = 0
    with torch.no_grad():
        for index in range(min(limit, len(dataset))):
            sample = dataset[index]
            data = {key: sample[key].unsqueeze(0).to(device)
                    for key in ('text', 'audio', 'vision')}
            masks = {key: sample[key].unsqueeze(0).to(device)
                     for key in ('text_mask', 'audio_mask', 'vision_mask')}
            full = model(data['text'], data['audio'], data['vision'],
                         **masks, return_coalitions=True)
            for bits in range(1, 8):
                reduced = {key: value.clone() for key, value in masks.items()}
                for modality, name in enumerate(('text', 'audio', 'vision')):
                    if not bits & (1 << modality):
                        reduced[name + '_mask'].fill_(False)
                if not any(mask.any() for mask in reduced.values()):
                    continue
                independent = model(data['text'], data['audio'], data['vision'],
                                    **reduced)['output_logit'][0, 0]
                recorded = full['coalition_logits'][0, bits]
                difference = (independent - recorded).abs().item()
                max_difference = max(max_difference, difference)
                comparisons += 1
                if not torch.isfinite(independent) or difference > tolerance:
                    raise AssertionError(
                        f'Input-mask disagreement for sample {index}, '
                        f'coalition {bits}: {difference}')
    print(f'{comparisons} masked-input comparisons; '
          f'max score difference {max_difference:.9g}')


if __name__ == '__main__':
    main()
