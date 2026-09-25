import unittest

import torch

from trains.singleTask.model.DLF import attribute_coalitions, exact_shapley


class CoalitionAttributionTests(unittest.TestCase):
    def test_additive_game_and_efficiency(self):
        values = torch.tensor([[0., 1., 2., 3., 4., 5., 6., 7.]])
        effects = exact_shapley(values)
        self.assertTrue(torch.allclose(effects, torch.tensor([[1., 2., 4.]])))
        self.assertTrue(torch.allclose(effects.sum(dim=1), values[:, 7] - values[:, 0]))

    def test_pairwise_interaction_is_shared(self):
        values = torch.tensor([[0., 0., 0., 2., 0., 0., 0., 2.]])
        self.assertTrue(torch.allclose(exact_shapley(values),
                                       torch.tensor([[1., 1., 0.]])))

    def test_signed_effect_and_route_decomposition(self):
        total = torch.tensor([[0., .4, -.2, .2, .1, .5, -.1, .3]])
        fixed = torch.tensor([[0., .3, -.2, .1, .1, .4, -.1, .2]])
        result = attribute_coalitions(total, fixed,
                                      torch.ones(1, 3, dtype=torch.bool),
                                      -.1, .2)
        self.assertEqual(result['decision_main_modality'].item(), 0)
        self.assertLess(result['shapley_decision_support'][0, 1].item(), 0)
        self.assertTrue(torch.allclose(
            result['shapley_content_support'] + result['shapley_route_effect'],
            result['shapley_decision_support']))
        self.assertTrue(result['full_modality_class_changed'][0, 0].item())

    def test_unavailable_modality_has_zero_contribution(self):
        values = torch.tensor([[0., .4, .1, .5, 0., .4, .1, .5]])
        result = attribute_coalitions(
            values, values, torch.tensor([[True, True, False]]), -.1, .2)
        self.assertEqual(result['shapley_decision_support'][0, 2].item(), 0)
        self.assertNotEqual(result['decision_main_modality'].item(), 2)

    def test_no_positive_winner_is_uncertain(self):
        values = torch.tensor([[0., .05, .05, .1, .05, .1, .1, .15]])
        result = attribute_coalitions(
            values, values, torch.ones(1, 3, dtype=torch.bool), -.1, .2)
        self.assertFalse(result['main_supportive'].item())
        self.assertTrue(result['main_uncertain'].item())
        self.assertEqual(result['positive_support_share'].sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
