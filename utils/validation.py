"""Select gate temperature and three-class thresholds on attachment 2 validation.

This script only supports the final coalition-aware model. The same validation
split selects temperature and thresholds, so reported metrics are not held-out
test estimates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.DLF import DLF, attribute_coalitions


NAMES = np.asarray(('text', 'audio', 'vision'))


def class_metrics(predicted, true):
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (true, predicted), 1)
    totals = confusion.sum(axis=1) + confusion.sum(axis=0)
    f1 = np.divide(2 * np.diag(confusion), totals,
                   out=np.zeros(3, dtype=float), where=totals != 0)
    return {'macro_f1': float(f1.mean()),
            'accuracy': float(np.trace(confusion) / len(true)),
            'f1_per_class': f1.tolist(), 'confusion': confusion.tolist()}


def calibrate_thresholds(scores, labels):
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n = len(scores)
    prefix = np.zeros((n + 1, 3), dtype=np.int64)
    for cls in range(3):
        prefix[1:, cls] = np.cumsum(sorted_labels == cls)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores) > 0) + 1, n]
    totals = prefix[-1]
    best = None
    for i in boundaries[:-1]:
        js = boundaries[boundaries > i]
        tp0 = prefix[i, 0]
        tp1 = prefix[js, 1] - prefix[i, 1]
        tp2 = totals[2] - prefix[js, 2]
        counts = [np.full(len(js), i), js - i, n - js]
        f1s = [np.divide(2 * tp, totals[k] + counts[k],
                         out=np.zeros(len(js), dtype=float),
                         where=(totals[k] + counts[k]) != 0)
               for k, tp in enumerate([np.full(len(js), tp0), tp1, tp2])]
        macro = sum(f1s) / 3
        accuracy = (tp0 + tp1 + tp2) / n
        for pos in np.flatnonzero(macro == macro.max()):
            candidate = (float(macro[pos]), float(accuracy[pos]),
                         int(i), int(js[pos]))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    _, _, i, j = best

    def midpoint(split):
        if split == 0:
            return float(np.nextafter(sorted_scores[0], -np.inf))
        if split == n:
            return float(np.nextafter(sorted_scores[-1], np.inf))
        return float((sorted_scores[split - 1] + sorted_scores[split]) / 2)

    lower, upper = midpoint(i), midpoint(j)
    predicted = np.where(scores < lower, 0, np.where(scores > upper, 2, 1))
    return lower, upper, class_metrics(predicted, labels), predicted


def evaluate(gate_temperature):
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / 'checkpoints/final/model.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args = get_config_regression('DLF', 'mosei')
    args.adaptive_main = True
    args.coalition_aware = True
    args.gate_temperature = gate_temperature
    args.counterfactual_specific_only = True
    args.feature_T = args.feature_A = args.feature_V = ''
    loader = MMDataLoader(args, num_workers=0, include_test=False)['valid']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DLF(args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    indices, scores, truths, weights = [], [], [], []
    coalition, fixed, availability = [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch['text'].to(device), batch['audio'].to(device),
                batch['vision'].to(device),
                text_mask=batch['text_mask'].to(device),
                audio_mask=batch['audio_mask'].to(device),
                vision_mask=batch['vision_mask'].to(device),
                return_coalitions=True)
            indices.extend(batch['index'].tolist())
            scores.extend(output['output_logit'].view(-1).cpu().tolist())
            truths.extend(batch['labels']['M'].view(-1).tolist())
            weights.extend(output['role_weights'].cpu().tolist())
            coalition.extend(output['coalition_logits'].cpu().tolist())
            fixed.extend(output['fixed_coalition_logits'].cpu().tolist())
            availability.extend(torch.stack([
                output['masks'][name].any(dim=1)
                for name in ('l', 'a', 'v')], dim=1).cpu().tolist())
    scores, truths = np.asarray(scores), np.asarray(truths)
    weights, availability = np.asarray(weights), np.asarray(availability,
                                                           dtype=bool)
    coalition, fixed = np.asarray(coalition), np.asarray(fixed)
    if not np.allclose(coalition[:, 7], scores, atol=1e-5, rtol=1e-5):
        raise AssertionError('Full coalition does not match predictor output.')
    classes = np.where(truths < 0, 0, np.where(truths > 0, 2, 1))
    lower, upper, metrics, predicted = calibrate_thresholds(scores, classes)
    audit = attribute_coalitions(
        torch.as_tensor(coalition), torch.as_tensor(fixed),
        torch.as_tensor(availability), lower, upper)
    audit = {key: value.cpu().numpy() for key, value in audit.items()}
    main = audit['decision_main_modality']
    gate_main = weights.argmax(axis=1)
    removal = audit['full_modality_ablation_scores']
    label_gain = np.abs(removal - truths[:, None]) - np.abs(
        scores[:, None] - truths[:, None])
    observed = np.where(availability, label_gain, -np.inf)
    oracle = observed.argmax(axis=1)
    ranked = np.sort(observed, axis=1)
    clear = ((availability.sum(axis=1) >= 2) & (ranked[:, -1] >= .02) &
             (ranked[:, -1] - ranked[:, -2] >= .02))
    after_class = np.where(removal < lower, 0,
                           np.where(removal > upper, 2, 1))
    selected = np.arange(len(scores))
    without_correct = after_class[selected, main] == classes
    full_correct = predicted == classes
    metrics.update({
        'model': 'coalition_route', 'n_valid': len(scores),
        'gate_temperature': gate_temperature,
        'threshold_negative_neutral': lower,
        'threshold_neutral_positive': upper,
        'mae': float(np.mean(np.abs(scores - truths))),
        'correlation': float(np.corrcoef(scores, truths)[0, 1]),
        'source': 'attachment_2_validation_only',
        'note': 'Thresholds and reported classification metrics use the same '
                'validation set; they are not held-out test estimates.',
        'gate_main_counts': {name: int((gate_main == i).sum())
                             for i, name in enumerate(NAMES)},
        'gate_mean_weights': weights.mean(axis=0).tolist(),
        'coalition_audit': {
            'definition': 'exact Shapley of full-model predicted-class margin '
                          'over all eight whole-modality coalitions',
            'not_ground_truth_modality_importance': True,
            'empty_coalition_score': 0.0,
            'main_counts': {name: int((main == i).sum())
                            for i, name in enumerate(NAMES)},
            'n_supportive': int(audit['main_supportive'].sum()),
            'n_uncertain': int(audit['main_uncertain'].sum()),
            'n_selected_class_flip': int(audit['main_class_critical'].sum()),
            'n_selected_correct_to_wrong': int((full_correct & ~without_correct).sum()),
            'n_selected_wrong_to_correct': int((~full_correct & without_correct).sum()),
            'selected_mean_label_error_increment': float(label_gain[selected, main].mean()),
            'always_text_mean_label_error_increment': float(label_gain[:, 0].mean()),
            'n_clear_label_benefit': int(clear.sum()),
            'match_label_benefit_oracle_clear': float((main[clear] == oracle[clear]).mean())
                                               if clear.any() else None,
            'always_text_match_label_benefit_oracle_clear': float((oracle[clear] == 0).mean())
                                                           if clear.any() else None,
            'gate_match_label_benefit_oracle_clear': float((gate_main[clear] == oracle[clear]).mean())
                                                   if clear.any() else None,
            'fusion_gate_agreement': float((main == gate_main).mean()),
            'mean_absolute_route_mediation': np.abs(audit['shapley_route_effect']).mean(axis=0).tolist(),
        },
    })
    columns = {'sample_index': indices, 'true_score': truths,
               'predicted_score': scores, 'true_class': classes,
               'predicted_class': predicted}
    for i, name in enumerate(NAMES):
        columns[f'weight_{name}'] = weights[:, i]
    columns.update({'fusion_main_modality': NAMES[gate_main],
                    'main_modality': NAMES[main],
                    'main_supportive': audit['main_supportive'],
                    'main_uncertain': audit['main_uncertain'],
                    'main_class_critical': audit['main_class_critical'],
                    'support_gap': audit['support_gap']})
    for i, name in enumerate(NAMES):
        columns[f'shapley_support_{name}'] = audit['shapley_decision_support'][:, i]
        columns[f'fixed_weight_content_{name}'] = audit['shapley_content_support'][:, i]
        columns[f'route_mediated_{name}'] = audit['shapley_route_effect'][:, i]
        columns[f'positive_share_{name}'] = audit['positive_support_share'][:, i]
        columns[f'label_error_increment_without_{name}'] = np.where(
            availability[:, i], label_gain[:, i], np.nan)
        columns[f'class_changed_without_{name}'] = audit['full_modality_class_changed'][:, i]
        columns[f'predicted_class_without_{name}'] = np.where(
            availability[:, i], after_class[:, i], -1)
    return metrics, pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate-temperatures', nargs='+', type=float,
                        default=[1.0])
    parser.add_argument('--mae-slack', type=float, default=0.005)
    cli = parser.parse_args()
    if not cli.gate_temperatures or any(t <= 0 for t in cli.gate_temperatures):
        parser.error('Gate temperatures must be positive.')
    if cli.mae_slack < 0:
        parser.error('MAE slack must be nonnegative.')
    outcomes = [evaluate(t) for t in cli.gate_temperatures]
    best_mae = min(metrics['mae'] for metrics, _ in outcomes)
    eligible = [item for item in outcomes
                if item[0]['mae'] <= best_mae + cli.mae_slack]
    metrics, frame = max(eligible, key=lambda item:
                         (item[0]['macro_f1'], -item[0]['mae']))
    metrics['gate_temperature_candidates'] = cli.gate_temperatures
    metrics['gate_selection_mae_slack'] = cli.mae_slack
    metrics['gate_selection_metric'] = 'macro_f1_with_mae_constraint'
    out = Path(__file__).resolve().parents[1] / 'outputs/validation'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'valid_predictions.csv', index=False)
    with (out / 'valid_thresholds.json').open('w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
