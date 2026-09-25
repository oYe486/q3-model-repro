"""Independently recheck all 20 attachment-4 prediction/evidence records.

This is an algorithmic provenance and score audit. It never promotes an ASR
candidate to a human/feature-extraction-verified media location. It also checks
decoded video bounds and the input/evidence error-attribution flags.
"""

import json
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import BertTokenizerFast

from infer_attachment4 import (MIN_MEDIA_TEXT_MATCH, MODEL_TOKEN_LIMIT,
                               _batch, _class, _error_attribution,
                               _expanded_char_span, _score,
                               _video_decode_info)
from .evidence_windows import NAMES, class_margin, occlude
from .attachment4_data import load_aligned_sample
from .evidence_location import EvidenceLocation
from .media_alignment import span_time
from trains.singleTask.model.DLF import DLF


def _close(actual, expected, tolerance=1e-5):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f'Expected {expected}, got {actual}.')


def verify():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / 'outputs/attachment4'
    records = [json.loads(line) for line in (
        output_dir / 'predictions_20.jsonl').open(encoding='utf-8')]
    ids = [record['sample_id'] for record in records]
    if ids != [f'{i:02d}' for i in range(1, 21)]:
        raise AssertionError('Prediction IDs are not exactly 01 through 20.')
    tokenizer = BertTokenizerFast.from_pretrained(
        str(root / 'pretrained/bert-base-uncased'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DLF.from_final_checkpoint(root=root, device=device)
    audit = []
    with torch.no_grad():
        for record in records:
            sample_id = record['sample_id']
            sample = load_aligned_sample(record['feature_file'])
            if (sample['id'] != sample_id or
                    sample['video_file'] != record['video_file'] or
                    sample['raw_text'] != record['raw_text']):
                raise AssertionError(f'{sample_id}: source provenance mismatch.')
            base = _batch(sample, device)
            output = model(
                base['text'], base['audio'], base['vision'],
                text_mask=base['text_mask'],
                audio_mask=base['audio_mask'],
                vision_mask=base['vision_mask'],
                return_explanation=True)
            score = float(output['output_logit'][0, 0])
            _close(score, record['predicted_score'])
            cls = int(output['predicted_class'][0])
            if cls != record['predicted_class']:
                raise AssertionError(f'{sample_id}: class changed on replay.')
            main = NAMES[int(output['main_modality'][0])]
            if main != record['main_modality']:
                raise AssertionError(f'{sample_id}: main changed on replay.')
            for effect_name, model_key in (
                    ('modality_signed_support', 'shapley_decision_support'),
                    ('fixed_weight_content_effect', 'shapley_content_support'),
                    ('route_mediated_effect', 'shapley_route_effect')):
                for i, name in enumerate(NAMES):
                    _close(float(output[model_key][0, i]),
                           record[effect_name][name])
            encoded = tokenizer(
                sample['raw_text'], max_length=50, truncation=True,
                padding='max_length', return_offsets_mapping=True)
            if encoded['input_ids'] != sample['text'][0].long().tolist():
                raise AssertionError(f'{sample_id}: BERT offsets disagree.')
            cache = output_dir / 'asr' / f'{sample_id}_base.en.json'
            with cache.open(encoding='utf-8') as stream:
                alignment = json.load(stream)['alignment']
            alignment['raw_to_asr'] = {
                int(key): value for key, value in
                alignment['raw_to_asr'].items()}
            _close(alignment['raw_word_match_rate'],
                   record['asr_audit']['raw_word_match_rate'])
            video_info = _video_decode_info(sample['video_file'])
            for key, expected in (
                    ('video_duration_seconds', video_info['duration_seconds']),
                    ('video_reported_duration_seconds',
                     video_info['reported_duration_seconds']),
                    ('video_reported_frame_count',
                     video_info['reported_frame_count']),
                    ('video_decoded_frame_count',
                     video_info['decoded_frame_count'])):
                _close(record['asr_audit'][key], expected)
            raw_token_count = len(tokenizer(sample['raw_text'])['input_ids'])
            expected_flags = _error_attribution(
                sample, raw_token_count, alignment, video_info)
            if record['error_attribution'] != expected_flags:
                raise AssertionError(f'{sample_id}: error attribution changed.')
            if abs(record['asr_audit']['audio_duration_seconds'] -
                   video_info['duration_seconds']) <= .25 and any(
                    'decoded audio/video duration mismatch' in warning
                    for warning in record['warnings']):
                raise AssertionError('False audio/video duration warning.')
            counts = {'text_exact': 0, 'media_candidate': 0,
                      'feature_only': 0, 'vision_frame': 0}
            for name in NAMES:
                for evidence in record['top_evidence'][name]:
                    location = EvidenceLocation(**evidence['location'])
                    location.validate(raw_text=sample['raw_text'])
                    if location.sample_id != sample_id or location.modality != name:
                        raise AssertionError('Evidence source ID/modality mismatch.')
                    if (location.source_feature_file != sample['feature_file'] or
                            location.source_video_file != sample['video_file'] or
                            location.feature_length != 50):
                        raise AssertionError('Evidence source path/length mismatch.')
                    span = _expanded_char_span(
                        encoded['offset_mapping'], location.feature_start,
                        location.feature_end_exclusive, sample['raw_text'])
                    fragment = (sample['raw_text'][span[0]:span[1]]
                                if span is not None else None)
                    if fragment != location.aligned_transcript_fragment:
                        raise AssertionError('Evidence fragment differs from BERT offsets.')
                    if name == 'text':
                        if (location.mapping_status != 'text_exact' or
                                span != (location.raw_text_char_start,
                                         location.raw_text_char_end_exclusive)):
                            raise AssertionError('Text evidence is not exactly located.')
                    elif location.mapping_status == 'media_candidate':
                        if alignment['raw_word_match_rate'] < MIN_MEDIA_TEXT_MATCH:
                            raise AssertionError('Media mapping must be withheld on text/media mismatch.')
                        expected, _ = span_time(alignment, span[0], span[1])
                        if expected is None:
                            raise AssertionError('Claimed media time lacks exact ASR words.')
                        _close(location.media_start_seconds, expected[0])
                        _close(location.media_end_seconds, expected[1])
                        if not 0 <= expected[0] < expected[1] <= (
                                alignment['audio_duration_seconds'] + .05):
                            raise AssertionError('Audio time outside source stream.')
                        if expected[1] > video_info['duration_seconds'] + .05:
                            raise AssertionError('Media time exceeds decodable video.')
                    else:
                        if location.mapping_status != 'feature_only':
                            raise AssertionError('Unexpected localization status.')
                    if location.video_frame_index is not None:
                        if location.video_frame_index >= video_info['decoded_frame_count']:
                            raise AssertionError('Visual frame exceeds decodable stream.')
                        capture = cv2.VideoCapture(sample['video_file'])
                        capture.set(cv2.CAP_PROP_POS_FRAMES,
                                    location.video_frame_index)
                        okay, _ = capture.read()
                        actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        actual_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                        capture.release()
                        if not okay or actual_index != location.video_frame_index:
                            raise AssertionError('Visual frame failed independent decode.')
                        _close(actual_time,
                               location.video_frame_time_seconds, tolerance=.002)
                        counts['vision_frame'] += 1
                    counts[location.mapping_status] += 1
            primary = record['primary_evidence']
            if primary != record['top_evidence'][main][0]:
                raise AssertionError('Primary evidence is not main-modality top.')
            audio_preview = record['primary_audio_preview_file']
            if main == 'audio' and primary['location'][
                    'mapping_status'] == 'media_candidate':
                expected_path = output_dir / 'audio_primary' / f'{sample_id}.wav'
                if audio_preview != str(expected_path):
                    raise AssertionError('Primary audio preview path mismatch.')
                with wave.open(str(expected_path), 'rb') as stream:
                    if stream.getframerate() != 16000 or stream.getnchannels() != 1:
                        raise AssertionError('Primary audio preview format mismatch.')
                    duration = stream.getnframes() / stream.getframerate()
                interval = primary['location']
                _close(duration, interval['media_end_seconds'] -
                       interval['media_start_seconds'], tolerance=.03)
            elif audio_preview is not None:
                raise AssertionError('Unexpected primary audio preview.')
            variant = occlude(base, main, primary['start'],
                              primary['end_exclusive'])
            changed = float(_score(model, variant)[0])
            _close(changed, primary['occluded_score'])
            support = (class_margin(score, cls, model.lower, model.upper) -
                       class_margin(changed, cls, model.lower, model.upper))
            _close(support, primary['decision_support'])
            if bool(primary['class_changed']) != (
                    _class(changed, model.lower, model.upper) != cls):
                raise AssertionError('Primary class flip disagrees with replay.')
            if sample_id == '13' and (
                    sample['vision_mask'].any() or record['top_evidence']['vision']):
                raise AssertionError('Sample 13 must not claim visual evidence.')
            flag_codes = {flag['code'] for flag in record['error_attribution']}
            if sample_id in ('07', '18') and (
                    raw_token_count <= MODEL_TOKEN_LIMIT or
                    'raw_text_truncated_by_model' not in flag_codes):
                raise AssertionError('Long text must be marked as truncated.')
            if sample_id == '15' and (
                    'text_media_inconsistent' not in flag_codes or
                    counts['media_candidate'] or counts['vision_frame']):
                raise AssertionError('Sample 15 must not claim media localization.')
            preview = output_dir / 'frames' / f'{sample_id}_vision.jpg'
            top_visual = record['top_evidence']['vision'][:1]
            expected_index = (top_visual[0]['location']['video_frame_index']
                              if top_visual else None)
            if expected_index is None:
                if preview.exists():
                    raise AssertionError('Stale visual preview has no current frame.')
            else:
                preview_image = cv2.imread(str(preview))
                capture = cv2.VideoCapture(sample['video_file'])
                capture.set(cv2.CAP_PROP_POS_FRAMES, expected_index)
                okay, source_image = capture.read()
                capture.release()
                if not okay or preview_image is None or (
                        preview_image.shape != source_image.shape):
                    raise AssertionError('Visual preview cannot be replayed.')
                image_error = np.abs(preview_image.astype(np.float32) -
                                     source_image.astype(np.float32)).mean()
                if image_error > 5:
                    raise AssertionError('Visual preview differs from source frame.')
            audit.append({
                'sample_id': sample_id,
                'prediction_replayed': True,
                'shapley_replayed': True,
                'primary_occlusion_replayed': True,
                'all_locations_contract_checked': True,
                'location_counts': counts,
                'error_attribution_codes': sorted(flag_codes),
                'primary_status': primary['location']['mapping_status'],
                'media_requires_extraction_provenance_review': True,
                'warnings': record['warnings'],
            })
            print(f'{sample_id}: PASS, primary={main}, '
                  f'{primary["location"]["mapping_status"]}, '
                  f'frames={counts["vision_frame"]}', flush=True)
    result = {
        'sample_count': len(audit),
        'all_algorithmic_checks_passed': True,
        'media_candidates_are_not_feature_provenance_verified': True,
        'samples': audit,
    }
    with (output_dir / 'verification_20.json').open('w',
                                                     encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print('All 20 samples passed replay and provenance-contract checks.')


if __name__ == '__main__':
    verify()
