import unittest
from pathlib import Path

from infer_attachment4 import _error_attribution, _video_decode_info
from utils.attachment4_data import load_aligned_sample
from utils.evidence_location import EvidenceLocation


ROOT = (Path(__file__).resolve().parents[2] /
        'datasets/附件4-可解释专项视频样本与特征文件/对齐版本')


class Attachment4ContractTests(unittest.TestCase):
    def test_aligned_adapter_is_label_free_and_matches_training_shapes(self):
        sample = load_aligned_sample(ROOT / '01.pkl')
        self.assertEqual(tuple(sample['text'].shape), (3, 50))
        self.assertEqual(tuple(sample['audio'].shape), (50, 74))
        self.assertEqual(tuple(sample['vision'].shape), (50, 35))
        self.assertEqual(tuple(sample['text_mask'].shape), (50,))
        self.assertNotIn('labels', sample)
        self.assertTrue(Path(sample['video_file']).is_file())

    def test_unresolved_media_cannot_claim_seconds(self):
        location = EvidenceLocation(
            sample_id='01', feature_version='aligned', modality='audio',
            feature_start=2, feature_end_exclusive=5, feature_length=50,
            source_feature_file='01.pkl', source_video_file='01.mp4',
            media_start_seconds=1.0, media_end_seconds=1.5)
        with self.assertRaises(ValueError):
            location.validate()

    def test_unaligned_features_cannot_enter_aligned_adapter(self):
        with self.assertRaises(ValueError):
            load_aligned_sample(ROOT.parent / '未对齐版本' / '01.pkl')

    def test_exact_raw_text_fragment_is_checked(self):
        location = EvidenceLocation(
            sample_id='01', feature_version='aligned', modality='text',
            feature_start=1, feature_end_exclusive=2, feature_length=50,
            source_feature_file='01.pkl', source_video_file='01.mp4',
            mapping_status='text_exact', raw_text_char_start=0,
            raw_text_char_end_exclusive=5, raw_text_fragment='Hello')
        self.assertEqual(location.to_dict(raw_text='Hello there')['raw_text_fragment'],
                         'Hello')
        with self.assertRaises(ValueError):
            location.validate(raw_text='Other text')

    def test_verified_visual_frame_requires_time_index_and_review(self):
        location = EvidenceLocation(
            sample_id='01', feature_version='aligned', modality='vision',
            feature_start=2, feature_end_exclusive=5, feature_length=50,
            source_feature_file='01.pkl', source_video_file='01.mp4',
            mapping_status='media_verified', mapping_method='manual_video_review',
            media_start_seconds=1.0, media_end_seconds=1.5,
            video_frame_time_seconds=1.25)
        with self.assertRaises(ValueError):
            location.validate()
        location.video_frame_index = 30
        location.verification_note = 'Reviewed against the source video.'
        location.validate()

    def test_video_duration_uses_decoded_frames_not_container_metadata(self):
        info = _video_decode_info(ROOT / 'videos' / '09.mp4')
        self.assertEqual(info['reported_frame_count'], 339)
        self.assertEqual(info['decoded_frame_count'], 151)
        self.assertAlmostEqual(info['duration_seconds'], 5.033333, places=3)

    def test_error_attribution_flags_known_aligned_issues(self):
        sample = load_aligned_sample(ROOT / '13.pkl')
        flags = _error_attribution(
            sample, 32, {'raw_word_match_rate': .93},
            {'reported_frame_count': 260, 'decoded_frame_count': 260})
        self.assertEqual([flag['code'] for flag in flags],
                         ['aligned_visual_features_unavailable'])
        sample = load_aligned_sample(ROOT / '15.pkl')
        flags = _error_attribution(
            sample, 23, {'raw_word_match_rate': .235},
            {'reported_frame_count': 220, 'decoded_frame_count': 220})
        self.assertIn('text_media_inconsistent',
                      [flag['code'] for flag in flags])


if __name__ == '__main__':
    unittest.main()
