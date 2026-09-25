"""Audio-transcript timing candidates from the supplied MP4, never index scaling.

Whisper is used as an independent recognizer. Only verbatim, monotonic matches
between its timestamped words and supplied raw_text receive candidate times.
This does not establish the provenance of pre-extracted aligned features; all
times remain candidates until the media and feature extraction are reviewed.
"""

import hashlib
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
SAMPLE_RATE = 16000
ASR_MODEL = 'tiny.en'


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _normalize(word):
    return ''.join(character for character in word.lower().replace('’', "'")
                   if character.isalnum())


def decode_audio(video_path):
    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner',
               '-loglevel', 'error', '-i', str(video_path), '-vn', '-f',
               's16le', '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1']
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode or not process.stdout:
        raise ValueError(f'Audio decoding failed for {video_path}: '
                         f'{process.stderr.decode("utf-8", errors="replace")[:300]}')
    return np.frombuffer(process.stdout, dtype='<i2').astype(np.float32) / 32768


def align_words(raw_text, asr_words):
    raw = [{'word': match.group(), 'start_char': match.start(),
            'end_char_exclusive': match.end(),
            'normalized': _normalize(match.group())}
           for match in WORD_RE.finditer(raw_text)]
    recognized = []
    for item in asr_words:
        normalized = _normalize(item['word'])
        if normalized and float(item['end']) > float(item['start']):
            recognized.append({
                'word': item['word'], 'normalized': normalized,
                'start_seconds': float(item['start']),
                'end_seconds': float(item['end']),
                'probability': float(item.get('probability', 0)),
            })
    matches = SequenceMatcher(None, [item['normalized'] for item in raw],
                              [item['normalized'] for item in recognized],
                              autojunk=False).get_matching_blocks()
    matched = {}
    for block in matches:
        for offset in range(block.size):
            matched[block.a + offset] = block.b + offset
    return {'raw_words': raw, 'asr_words': recognized,
            'raw_to_asr': matched,
            'raw_word_match_rate': len(matched) / max(len(raw), 1),
            'recognized_text': ' '.join(item['word'].strip()
                                        for item in recognized)}


def span_time(alignment, char_start, char_end):
    """Return a candidate interval only when every overlapping word matches."""
    words = alignment['raw_words']
    selected = [i for i, word in enumerate(words)
                if word['start_char'] < char_end and
                word['end_char_exclusive'] > char_start]
    if not selected:
        return None, 'No raw-text word overlaps the feature window.'
    mapping = alignment['raw_to_asr']
    missing = [words[i]['word'] for i in selected if i not in mapping]
    if missing:
        return None, 'ASR did not exactly match: ' + ', '.join(missing[:8])
    indices = [int(mapping[i]) for i in selected]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        return None, 'Matched words are not contiguous in the audio.'
    asr_words = alignment['asr_words']
    start = asr_words[indices[0]]['start_seconds']
    end = asr_words[indices[-1]]['end_seconds']
    if not 0 <= start < end:
        return None, 'Invalid ASR word time interval.'
    return (start, end), f'{len(selected)}/{len(selected)} exact matched words'


def transcribe_and_align(video_path, raw_text, model, cache_path=None,
                         model_name=ASR_MODEL):
    video_path = Path(video_path)
    cache_path = Path(cache_path) if cache_path is not None else None
    fingerprint = {'video_sha256': _digest(video_path),
                   'raw_text_sha256': hashlib.sha256(
                       raw_text.encode('utf-8')).hexdigest(),
                   'model': model_name, 'word_timestamps': True}
    if cache_path is not None and cache_path.is_file():
        with cache_path.open(encoding='utf-8') as stream:
            cached = json.load(stream)
        if cached.get('fingerprint') == fingerprint:
            cached['alignment']['raw_to_asr'] = {
                int(key): value for key, value in
                cached['alignment']['raw_to_asr'].items()}
            return cached['alignment']
    waveform = decode_audio(video_path)
    result = model.transcribe(
        waveform, language='en', task='transcribe', fp16=False,
        word_timestamps=True, temperature=0,
        condition_on_previous_text=False, verbose=False)
    words = [word for segment in result['segments']
             for word in segment.get('words', [])]
    alignment = align_words(raw_text, words)
    alignment['audio_duration_seconds'] = len(waveform) / SAMPLE_RATE
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('w', encoding='utf-8') as stream:
            json.dump({'fingerprint': fingerprint, 'alignment': alignment},
                      stream, ensure_ascii=False, indent=2)
    return alignment
