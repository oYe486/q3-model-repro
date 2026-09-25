"""Frozen question-3 inference and per-sample evidence audit for attachment 4.

Only aligned features enter the trained model. MP4 audio supplies independent
word-time *candidates*; actual MP4 frames are decoded at those candidate
times. No attachment-4 label or sample is used to change model parameters,
fusion structure, evidence ranking, or validation-selected thresholds.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import whisper
from transformers import BertTokenizerFast

from utils.evidence_windows import NAMES, candidate_windows, class_margin, occlude
from utils.attachment4_data import load_aligned_sample
from utils.evidence_location import EvidenceLocation
from utils.media_alignment import span_time, transcribe_and_align
from trains.singleTask.model.DLF import DLF

MODEL_TOKEN_LIMIT = 50
MIN_MEDIA_TEXT_MATCH = 0.70


def _batch(sample, device):
    return {name: sample[name].unsqueeze(0).to(device)
            for name in ('text', 'audio', 'vision', 'text_mask',
                         'audio_mask', 'vision_mask')}


def _score(model, batch):
    result = model(batch['text'], batch['audio'], batch['vision'],
                   text_mask=batch['text_mask'],
                   audio_mask=batch['audio_mask'],
                   vision_mask=batch['vision_mask'])
    return result['output_logit'].reshape(-1)


def _class(score, lower, upper):
    return 0 if score < lower else (2 if score > upper else 1)


def _expanded_char_span(offsets, start, end, raw_text):
    covered = [(left, right) for left, right in offsets[start:end]
               if right > left]
    if not covered:
        return None
    left = min(value[0] for value in covered)
    right = max(value[1] for value in covered)
    while left > 0 and not raw_text[left - 1].isspace():
        left -= 1
    while right < len(raw_text) and not raw_text[right].isspace():
        right += 1
    return left, right


def _video_decode_info(video_path):
    """Measure the decodable stream instead of trusting MP4 frame metadata."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f'Video cannot be decoded: {video_path}')
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid video FPS: {video_path}')
        decoded_count = 0
        last_time = None
        while True:
            okay, _ = capture.read()
            if not okay:
                break
            decoded_count += 1
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if np.isfinite(timestamp) and timestamp >= 0:
                last_time = timestamp
        if decoded_count == 0:
            raise ValueError(f'Video has no decodable frame: {video_path}')
        duration = (last_time + 1 / fps if last_time is not None and
                    (last_time > 0 or decoded_count == 1)
                    else decoded_count / fps)
        return {
            'fps': fps,
            'reported_frame_count': reported_count,
            'decoded_frame_count': decoded_count,
            'reported_duration_seconds': reported_count / fps,
            'duration_seconds': duration,
        }
    finally:
        capture.release()


def _error_attribution(sample, raw_token_count, alignment, video_info):
    """Describe input/coverage limits without changing any supplied sample."""
    flags = []
    if not bool(sample['vision_mask'].any()):
        flags.append({
            'code': 'aligned_visual_features_unavailable',
            'origin': 'supplied_aligned_features',
            'effect': 'vision excluded; no model-grounded visual keyframe',
        })
    if raw_token_count > MODEL_TOKEN_LIMIT:
        flags.append({
            'code': 'raw_text_truncated_by_model',
            'origin': 'model_input_limit',
            'raw_bert_tokens': raw_token_count,
            'modeled_bert_tokens': MODEL_TOKEN_LIMIT,
            'effect': 'evidence limited to the modeled text prefix',
        })
    if alignment['raw_word_match_rate'] < MIN_MEDIA_TEXT_MATCH:
        flags.append({
            'code': 'text_media_inconsistent',
            'origin': 'supplied_text_vs_source_media',
            'raw_word_match_rate': alignment['raw_word_match_rate'],
            'effect': 'media time/frame mapping withheld; text spans refer only to supplied raw_text',
        })
    if video_info['reported_frame_count'] != video_info['decoded_frame_count']:
        flags.append({
            'code': 'video_frame_metadata_inaccurate',
            'origin': 'source_video_metadata',
            'reported_frames': video_info['reported_frame_count'],
            'decoded_frames': video_info['decoded_frame_count'],
            'effect': 'actual decoded duration used for evidence checks',
        })
    return flags


def _decode_frame(video_path, interval, video_info):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f'Video cannot be decoded: {video_path}')
    try:
        fps = video_info['fps']
        count = video_info['decoded_frame_count']
        duration = video_info['duration_seconds']
        if interval[0] >= duration:
            return {'index': None, 'seconds': None, 'fps': fps,
                    'frame_count': count, 'duration_seconds': duration,
                    'image': None}
        target = (interval[0] + interval[1]) / 2
        index = max(0, min(count - 1, int(round(target * fps))))
        candidates = [index + delta for delta in
                      (0, -1, 1, -2, 2, -3, 3, -5, 5)]
        for frame_index in candidates:
            if not 0 <= frame_index < count:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            okay, frame = capture.read()
            if not okay:
                continue
            actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            actual_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if (np.isfinite(actual_time) and
                    interval[0] <= actual_time <= interval[1]):
                return {'index': actual_index, 'seconds': actual_time,
                        'fps': fps, 'frame_count': count,
                        'duration_seconds': duration, 'image': frame}
        return {'index': None, 'seconds': None, 'fps': fps,
                'frame_count': count, 'duration_seconds': duration,
                'image': None}
    finally:
        capture.release()


def _top_windows(model, base, full_score, predicted_class, lower, upper,
                 window_size, top_k, batch_size):
    masks = {name: base[name + '_mask'] for name in NAMES}
    windows = candidate_windows(masks, window_size)
    descriptions = [(name, start, end) for name in NAMES
                    for start, end in windows[name]]
    if not descriptions:
        raise ValueError('No observed evidence windows.')
    changed_scores = []
    for offset in range(0, len(descriptions), batch_size):
        current = descriptions[offset:offset + batch_size]
        variants = [occlude(base, name, start, end)
                    for name, start, end in current]
        combined = {key: torch.cat([variant[key] for variant in variants])
                    for key in base}
        changed_scores.extend(_score(model, combined).cpu().tolist())
    original_margin = class_margin(full_score, predicted_class, lower, upper)
    results = {name: [] for name in NAMES}
    for (name, start, end), changed in zip(descriptions, changed_scores):
        support = original_margin - class_margin(
            changed, predicted_class, lower, upper)
        results[name].append({
            'start': start, 'end_exclusive': end,
            'occluded_score': float(changed),
            'decision_support': float(support),
            'absolute_score_change': abs(full_score - changed),
            'class_changed': _class(changed, lower, upper) != predicted_class,
        })
    for name in NAMES:
        items = results[name]
        if any(item['decision_support'] > 0 for item in items):
            items.sort(key=lambda item: item['decision_support'], reverse=True)
        else:
            items.sort(key=lambda item: item['absolute_score_change'], reverse=True)
        results[name] = items[:top_k]
    return results


def _render_contact_sheet(frame_files, output_path):
    tiles = []
    for sample_id, path, caption in frame_files:
        tile = np.full((220, 360, 3), 235, dtype=np.uint8)
        if path is not None:
            frame = cv2.imread(str(path))
            if frame is not None:
                scaled = cv2.resize(frame, (360, 190))
                tile[:190] = scaled
        cv2.putText(tile, f'{sample_id}: {caption}'[:49], (7, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.full((220, 360, 3), 235, dtype=np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(output_path), np.vstack(rows))


def _save_audio_excerpt(video_path, interval, target):
    import imageio_ffmpeg

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner',
               '-loglevel', 'error', '-y', '-i', str(video_path),
               '-ss', str(interval[0]), '-to', str(interval[1]),
               '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
               str(target)]
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode or not target.is_file() or target.stat().st_size == 0:
        raise ValueError('Failed to extract primary audio preview: '
                         + process.stderr.decode('utf-8', errors='replace')[:300])


def analyze(args):
    root = Path(__file__).parent
    data_root = (root.parent / 'datasets' /
                 '附件4-可解释专项视频样本与特征文件' / '对齐版本')
    paths = sorted(data_root.glob('*.pkl'))
    if len(paths) != 20:
        raise ValueError(f'Expected 20 aligned attachment-4 samples, got {len(paths)}.')
    if args.limit < 1 or args.limit > len(paths):
        raise ValueError('limit must be between 1 and 20.')
    out = root / 'outputs/attachment4'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'frames').mkdir(exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(
        str(root / 'pretrained/bert-base-uncased'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DLF.from_final_checkpoint(root=root, device=device)
    asr = whisper.load_model(args.asr_model, device=str(device))
    records, previews = [], []
    with torch.no_grad():
        for feature_path in paths[:args.limit]:
            sample = load_aligned_sample(feature_path)
            base = _batch(sample, device)
            original = model(base['text'], base['audio'], base['vision'],
                             text_mask=base['text_mask'],
                             audio_mask=base['audio_mask'],
                             vision_mask=base['vision_mask'],
                             return_explanation=True)
            score = float(original['output_logit'][0, 0])
            lower, upper = model.lower, model.upper
            predicted_class = int(original['predicted_class'][0])
            if predicted_class != _class(score, lower, upper):
                raise AssertionError('Prediction class and calibrated score disagree.')
            local = _top_windows(model, base, score,
                                 predicted_class, lower, upper,
                                 args.window_size, args.top_k,
                                 args.batch_size)
            raw_text = sample['raw_text']
            tokenized = tokenizer(
                raw_text, max_length=MODEL_TOKEN_LIMIT, truncation=True,
                padding='max_length', return_offsets_mapping=True)
            if tokenized['input_ids'] != sample['text'][0].long().tolist():
                raise AssertionError(f'{sample["id"]}: text token alignment failed.')
            offsets = tokenized['offset_mapping']
            raw_token_count = len(tokenizer(raw_text)['input_ids'])
            alignment = transcribe_and_align(
                sample['video_file'], raw_text, asr,
                cache_path=out / 'asr' / f'{sample["id"]}_{args.asr_model}.json',
                model_name=args.asr_model)
            video_info = _video_decode_info(sample['video_file'])
            error_attribution = _error_attribution(
                sample, raw_token_count, alignment, video_info)
            media_mapping_allowed = (
                alignment['raw_word_match_rate'] >= MIN_MEDIA_TEXT_MATCH)
            if alignment['raw_word_match_rate'] < 0.5:
                print(f'{sample["id"]}: low ASR/raw text match '
                      f'{alignment["raw_word_match_rate"]:.2f}', flush=True)
            evidence = {}
            frame_preview = None
            visual_note = 'no visual evidence'
            for name in NAMES:
                evidence[name] = []
                for candidate in local[name]:
                    span = _expanded_char_span(
                        offsets, candidate['start'],
                        candidate['end_exclusive'], raw_text)
                    fragment = (raw_text[span[0]:span[1]]
                                if span is not None else None)
                    location = EvidenceLocation(
                        sample_id=sample['id'], feature_version='aligned',
                        modality=name, feature_start=candidate['start'],
                        feature_end_exclusive=candidate['end_exclusive'],
                        feature_length=50,
                        source_feature_file=sample['feature_file'],
                        source_video_file=sample['video_file'],
                        decision_support=candidate['decision_support'],
                        class_changed=candidate['class_changed'],
                        aligned_transcript_fragment=fragment)
                    mapping_note = None
                    if name == 'text' and span is not None:
                        location.mapping_status = 'text_exact'
                        location.raw_text_char_start = span[0]
                        location.raw_text_char_end_exclusive = span[1]
                        location.raw_text_fragment = fragment
                    elif name != 'text' and span is not None:
                        if media_mapping_allowed:
                            interval, mapping_note = span_time(
                                alignment, span[0], span[1])
                        else:
                            interval = None
                            mapping_note = (
                                'media mapping withheld: supplied text and '
                                'source audio have insufficient word agreement')
                        if interval is not None and (
                                interval[1] <= video_info['duration_seconds'] + .05):
                            location.mapping_status = 'media_candidate'
                            location.media_start_seconds = interval[0]
                            location.media_end_seconds = interval[1]
                            location.mapping_method = (
                                f'{args.asr_model}_word_timestamps+'
                                'exact_raw_word_match+bert_offset')
                            if name == 'vision':
                                frame = _decode_frame(
                                    sample['video_file'], interval, video_info)
                                if frame['index'] is not None:
                                    location.video_frame_index = frame['index']
                                    location.video_frame_time_seconds = frame['seconds']
                                    if len(evidence[name]) == 0:
                                        frame_preview = (out / 'frames' /
                                                         f'{sample["id"]}_vision.jpg')
                                        cv2.imwrite(str(frame_preview), frame['image'])
                                        visual_note = (f'frame {frame["index"]} '
                                                       f'@ {frame["seconds"]:.2f}s')
                        elif interval is not None:
                            mapping_note = (
                                'candidate interval exceeds the decodable '
                                'video duration; media mapping withheld')
                    record = dict(candidate)
                    record['location'] = location.to_dict(raw_text=raw_text)
                    record['time_mapping_note'] = mapping_note
                    evidence[name].append(record)
            main_index = int(original['main_modality'][0])
            main_name = NAMES[main_index]
            primary = evidence[main_name][0] if evidence[main_name] else None
            if primary is None:
                raise AssertionError('Main modality has no observed local evidence.')
            audio_preview = None
            if (main_name == 'audio' and primary['location'][
                    'mapping_status'] == 'media_candidate'):
                audio_preview = (out / 'audio_primary' /
                                 f'{sample["id"]}.wav')
                _save_audio_excerpt(
                    sample['video_file'],
                    (primary['location']['media_start_seconds'],
                     primary['location']['media_end_seconds']),
                    audio_preview)
            video_duration = video_info['duration_seconds']
            warnings = []
            if bool(original['main_uncertain'][0]):
                warnings.append('model-level main-modality separation is weak')
            if primary['decision_support'] < .02:
                warnings.append('primary local window has weak decision support (<0.02)')
            if abs(alignment['audio_duration_seconds'] - video_duration) > .25:
                warnings.append('decoded audio/video duration mismatch >0.25 seconds')
            if not media_mapping_allowed:
                warnings.append('text/media mismatch: media evidence mapping withheld')
            if raw_token_count > MODEL_TOKEN_LIMIT:
                warnings.append('raw text truncated to 50 model tokens')
            if not bool(sample['vision_mask'].any()):
                warnings.append('no observed visual feature; no visual keyframe claim')
            if video_info['reported_frame_count'] != video_info['decoded_frame_count']:
                warnings.append('video frame metadata inaccurate; decoded frame count used')
            for name in NAMES:
                if evidence[name] and evidence[name][0]['location'][
                        'mapping_status'] == 'feature_only':
                    warnings.append(f'{name} top evidence lacks verified time mapping')
            record = {
                'sample_id': sample['id'], 'feature_version': 'aligned',
                'raw_text': raw_text,
                'predicted_score': score,
                'predicted_class': predicted_class,
                'class_names': ['negative', 'neutral', 'positive'],
                'thresholds': {'negative_neutral': lower,
                               'neutral_positive': upper},
                'main_modality': main_name,
                'fusion_main_modality': NAMES[int(
                    original['fusion_main_modality'][0])],
                'main_supportive': bool(original['main_supportive'][0]),
                'main_uncertain': bool(original['main_uncertain'][0]),
                'main_class_critical': bool(original['main_class_critical'][0]),
                'modality_signed_support': dict(zip(NAMES, original[
                    'shapley_decision_support'][0].cpu().tolist())),
                'fixed_weight_content_effect': dict(zip(NAMES, original[
                    'shapley_content_support'][0].cpu().tolist())),
                'route_mediated_effect': dict(zip(NAMES, original[
                    'shapley_route_effect'][0].cpu().tolist())),
                'positive_support_share': dict(zip(NAMES, original[
                    'positive_support_share'][0].cpu().tolist())),
                'full_modality_class_changed': dict(zip(NAMES, original[
                    'full_modality_class_changed'][0].cpu().tolist())),
                'primary_evidence': primary,
                'primary_audio_preview_file': (str(audio_preview)
                                               if audio_preview is not None else None),
                'top_evidence': evidence,
                'error_attribution': error_attribution,
                'asr_audit': {
                    'model': args.asr_model,
                    'recognized_text': alignment['recognized_text'],
                    'raw_word_match_rate': alignment['raw_word_match_rate'],
                    'audio_duration_seconds': alignment['audio_duration_seconds'],
                    'video_duration_seconds': video_duration,
                    'video_reported_duration_seconds':
                        video_info['reported_duration_seconds'],
                    'video_reported_frame_count':
                        video_info['reported_frame_count'],
                    'video_decoded_frame_count':
                        video_info['decoded_frame_count'],
                },
                'warnings': warnings,
                'feature_file': sample['feature_file'],
                'video_file': sample['video_file'],
            }
            records.append(record)
            stale_preview = out / 'frames' / f'{sample["id"]}_vision.jpg'
            if frame_preview is None and stale_preview.is_file():
                # This exact generated preview may belong to an earlier ASR
                # run and must not masquerade as a current mapped frame.
                if stale_preview.resolve().parent != (out / 'frames').resolve():
                    raise AssertionError('Preview path left the output folder.')
                stale_preview.unlink()
            previews.append((sample['id'], frame_preview, visual_note))
            print(f'{sample["id"]}: {main_name}, {predicted_class}, '
                  f'primary={primary["location"]["mapping_status"]}, '
                  f'ASR={alignment["raw_word_match_rate"]:.2f}', flush=True)
    output_path = out / f'predictions_{len(records)}.jsonl'
    with output_path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    _render_contact_sheet(previews, out / 'visual_candidates_contact_sheet.jpg')
    statuses = {}
    for record in records:
        status = record['primary_evidence']['location']['mapping_status']
        statuses[status] = statuses.get(status, 0) + 1
    summary = {'n_samples': len(records), 'output_path': str(output_path),
               'primary_mapping_status_counts': statuses,
               'error_attribution_samples': {
                   code: [r['sample_id'] for r in records if any(
                       flag['code'] == code for flag in r['error_attribution'])]
                   for code in sorted({flag['code'] for r in records
                                       for flag in r['error_attribution']})},
               'no_visual_features': [r['sample_id'] for r in records
                                      if any('no observed visual' in warning
                                             for warning in r['warnings'])],
               'mean_asr_raw_word_match_rate': float(np.mean([
                   r['asr_audit']['raw_word_match_rate'] for r in records])),
               'all_media_times_are_candidates': True,
               'attachment4_used_for_training_or_selection': False}
    with (out / 'summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--window-size', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--asr-model', default='base.en')
    analyze(parser.parse_args())
