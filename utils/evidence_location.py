"""Provenance contract for later attachment-4 evidence localization.

Feature indices are not seconds or video-frame numbers. A localization record
may be promoted from feature_only to media_candidate or media_verified only
after a separate text/audio/video alignment step has supplied actual times.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Optional


@dataclass
class EvidenceLocation:
    sample_id: str
    feature_version: str              # aligned or unaligned
    modality: str                     # text, audio, or vision
    feature_start: int
    feature_end_exclusive: int
    feature_length: int
    source_feature_file: str
    source_video_file: str
    mapping_status: str = 'feature_only'
    decision_support: Optional[float] = None
    class_changed: Optional[bool] = None
    raw_text_char_start: Optional[int] = None
    raw_text_char_end_exclusive: Optional[int] = None
    raw_text_fragment: Optional[str] = None
    aligned_transcript_fragment: Optional[str] = None
    media_start_seconds: Optional[float] = None
    media_end_seconds: Optional[float] = None
    video_frame_index: Optional[int] = None
    video_frame_time_seconds: Optional[float] = None
    mapping_method: Optional[str] = None
    verification_note: Optional[str] = None

    def validate(self, raw_text=None):
        if not self.sample_id or self.feature_version not in ('aligned', 'unaligned'):
            raise ValueError('A sample ID and valid feature version are required.')
        if self.modality not in ('text', 'audio', 'vision'):
            raise ValueError('Unknown modality.')
        if not (0 <= self.feature_start < self.feature_end_exclusive <=
                self.feature_length):
            raise ValueError('Feature span is outside the sequence.')
        if not self.source_feature_file or not self.source_video_file:
            raise ValueError('Feature and video provenance are required.')
        if self.mapping_status not in ('feature_only', 'text_exact',
                                       'media_candidate', 'media_verified'):
            raise ValueError('Unknown localization status.')
        if self.decision_support is not None and not isfinite(self.decision_support):
            raise ValueError('Decision support must be finite.')
        char_fields = (self.raw_text_char_start, self.raw_text_char_end_exclusive,
                       self.raw_text_fragment)
        if any(value is not None for value in char_fields):
            if self.modality != 'text' or any(value is None for value in char_fields):
                raise ValueError('Exact raw-text spans belong only to text evidence.')
            if self.mapping_status != 'text_exact':
                raise ValueError('A raw-text span must be marked text_exact.')
            if not (0 <= self.raw_text_char_start < self.raw_text_char_end_exclusive):
                raise ValueError('Invalid raw-text character span.')
            if raw_text is not None and (
                raw_text[self.raw_text_char_start:self.raw_text_char_end_exclusive]
                    != self.raw_text_fragment):
                raise ValueError('Raw-text fragment does not match its source span.')
        time_fields = (self.media_start_seconds, self.media_end_seconds,
                       self.video_frame_time_seconds, self.video_frame_index)
        if self.mapping_status in ('feature_only', 'text_exact') and any(
                value is not None for value in time_fields):
            raise ValueError('Unresolved feature indices cannot claim media time.')
        if self.mapping_status == 'text_exact':
            if self.modality != 'text' or any(value is None for value in char_fields):
                raise ValueError('Text-exact status requires a matching text span.')
            if raw_text is None:
                raise ValueError('Text-exact status requires the source raw_text to verify.')
        if self.mapping_status in ('media_candidate', 'media_verified'):
            if self.modality == 'text' or not self.mapping_method:
                raise ValueError('Media localization requires a method and media modality.')
            if (self.media_start_seconds is None or self.media_end_seconds is None or
                    not isfinite(self.media_start_seconds) or
                    not isfinite(self.media_end_seconds) or
                    not 0 <= self.media_start_seconds < self.media_end_seconds):
                raise ValueError('A valid media time interval is required.')
        if self.video_frame_index is not None:
            if self.modality != 'vision' or self.video_frame_index < 0:
                raise ValueError('Frame indices belong only to vision evidence.')
        if self.video_frame_time_seconds is not None:
            if self.modality != 'vision' or not isfinite(self.video_frame_time_seconds):
                raise ValueError('Frame time belongs only to vision evidence.')
            if (self.media_start_seconds is not None and
                    not self.media_start_seconds <= self.video_frame_time_seconds <=
                    self.media_end_seconds):
                raise ValueError('Frame time must lie within the evidence interval.')
        if self.mapping_status == 'media_verified':
            if not self.verification_note:
                raise ValueError('Verified localization requires a review note.')
            if self.modality == 'vision' and (
                    self.video_frame_index is None or
                    self.video_frame_time_seconds is None):
                raise ValueError('A verified visual keyframe needs index and time.')
        return self

    def to_dict(self, raw_text=None):
        self.validate(raw_text=raw_text)
        return asdict(self)
