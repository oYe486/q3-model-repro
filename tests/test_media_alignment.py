import unittest

from utils.media_alignment import align_words, span_time


class MediaAlignmentTests(unittest.TestCase):
    def test_exact_monotone_words_get_candidate_time(self):
        aligned = align_words(
            "It's a terrible movie.", [
                {'word': " It's", 'start': 0., 'end': .3},
                {'word': ' a', 'start': .3, 'end': .4},
                {'word': ' terrible', 'start': .4, 'end': .8},
                {'word': ' movie', 'start': .8, 'end': 1.2},
            ])
        value, note = span_time(aligned, 7, 21)
        self.assertEqual(value, (.4, 1.2))
        self.assertIn('exact matched', note)

    def test_missing_word_is_not_interpolated(self):
        aligned = align_words(
            'one missing three', [
                {'word': ' one', 'start': 0., 'end': .3},
                {'word': ' three', 'start': .8, 'end': 1.2},
            ])
        value, note = span_time(aligned, 0, len('one missing three'))
        self.assertIsNone(value)
        self.assertIn('missing', note)

    def test_repeated_words_keep_monotone_order(self):
        aligned = align_words(
            'people often forgive people', [
                {'word': ' people', 'start': 0., 'end': .2},
                {'word': ' often', 'start': .2, 'end': .4},
                {'word': ' forgive', 'start': .4, 'end': .7},
                {'word': ' people', 'start': .7, 'end': 1.0},
            ])
        value, _ = span_time(aligned, 21, 27)
        self.assertEqual(value, (.7, 1.0))


if __name__ == '__main__':
    unittest.main()
