"""
here is the mian backbone for DLF
"""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   


def predicted_class(scores, lower, upper):
    """Map continuous scores to negative / neutral / positive (0 / 1 / 2)."""
    return torch.where(scores < lower, 0,
                       torch.where(scores > upper, 2, 1))


def decision_margin(scores, original_class, lower, upper):
    """Signed distance to the original class's nearest decision boundary."""
    return torch.where(original_class == 0, lower - scores,
                       torch.where(original_class == 2, scores - upper,
                                   torch.minimum(scores - lower,
                                                 upper - scores)))


def exact_shapley(values):
    """Compute exact T/A/V Shapley values for the eight coalition scores."""
    if values.ndim != 2 or values.size(1) != 8:
        raise ValueError('Expected eight coalition values per example.')
    result = []
    for modality in range(3):
        bit = 1 << modality
        terms = []
        for subset in range(8):
            if subset & bit:
                continue
            size = bin(subset).count('1')
            coefficient = (math.factorial(size) *
                           math.factorial(2 - size) / math.factorial(3))
            terms.append(coefficient * (values[:, subset | bit] -
                                        values[:, subset]))
        result.append(sum(terms))
    return torch.stack(result, dim=1)


def attribute_coalitions(coalition_scores, fixed_coalition_scores,
                         available, lower, upper,
                         minimum_support=0.02, minimum_gap=0.02):
    """Separate full decision support from fixed-route content support.

    Re-gated coalition Shapley is the reported model-internal contribution.
    Fixed original weights are zeroed for missing streams without
    renormalization; their difference is the controlled route effect, not a
    real-world causal effect or a label-error gain.
    """
    if coalition_scores.ndim != 2 or coalition_scores.size(1) != 8:
        raise ValueError('Expected eight coalition scores per example.')
    if fixed_coalition_scores.shape != coalition_scores.shape:
        raise ValueError('Fixed-route coalition shape mismatch.')
    if available.shape != (coalition_scores.size(0), 3):
        raise ValueError('Availability shape mismatch.')
    if not lower < upper or minimum_support <= 0 or minimum_gap <= 0:
        raise ValueError('Invalid thresholds.')
    available = available.bool()
    if not available.any(dim=1).all():
        raise ValueError('At least one modality is required.')
    if not torch.isfinite(coalition_scores).all() or not torch.isfinite(
            fixed_coalition_scores).all():
        raise ValueError('Coalition scores must be finite.')
    full_class = predicted_class(coalition_scores[:, 7], lower, upper)
    margins = decision_margin(coalition_scores, full_class[:, None], lower, upper)
    fixed_margins = decision_margin(fixed_coalition_scores,
                                    full_class[:, None], lower, upper)
    total = exact_shapley(margins)
    content = exact_shapley(fixed_margins)
    route = total - content
    score_effect = exact_shapley(coalition_scores)
    total = total.masked_fill(~available, 0)
    content = content.masked_fill(~available, 0)
    route = route.masked_fill(~available, 0)
    score_effect = score_effect.masked_fill(~available, 0)

    ranked = total.masked_fill(~available, -torch.inf)
    positive_main = ranked.argmax(dim=1)
    top = ranked.gather(1, positive_main[:, None]).squeeze(1)
    supportive = top > 0
    nominal = score_effect.abs().masked_fill(~available, -torch.inf).argmax(dim=1)
    main = torch.where(supportive, positive_main, nominal)
    positive = total.clamp_min(0)
    positive_share = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-12)
    top_two = ranked.topk(2, dim=1).values
    gap = torch.where(available.sum(dim=1) > 1,
                      top_two[:, 0] - top_two[:, 1],
                      torch.zeros_like(top_two[:, 0]))
    separated = (supportive & (available.sum(dim=1) > 1) &
                 (top >= minimum_support) & (gap >= minimum_gap))
    leave_one = torch.stack([coalition_scores[:, 7 ^ (1 << i)]
                             for i in range(3)], dim=1)
    leave_class = predicted_class(leave_one, lower, upper)
    changed = (leave_class != full_class[:, None]) & available
    return {
        'predicted_class': full_class,
        'coalition_margin': margins,
        'shapley_decision_support': total,
        'shapley_content_support': content,
        'shapley_route_effect': route,
        'shapley_score_effect': score_effect,
        'positive_support_share': positive_share,
        'decision_main_modality': main,
        'main_supportive': supportive,
        'main_uncertain': ~separated,
        'support_gap': gap,
        'full_modality_ablation_scores': leave_one,
        'full_modality_class_changed': changed,
        'main_class_critical': changed.gather(1, main[:, None]).squeeze(1),
    }

class DLF(nn.Module):
    def __init__(self, args):
        super(DLF, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads     
        self.layers = args.nlevels 
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.kernel_l = args.conv1d_kernel_size_l
        self.kernel_a = args.conv1d_kernel_size_a
        self.kernel_v = args.conv1d_kernel_size_v
        combined_dim_low = self.d_a    
        combined_dim_high = self.d_a 
        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        #   Modality-shared encoder 
        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)     
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)


        # 4 Multimodal Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers = self.layers)  
        self.trans_l_with_v = self.get_network(self_type='lv', layers = self.layers) 
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        
        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # A single dynamic-query fusion block replaces the three fixed LFA
        # branches when adaptive_main is enabled. The gate sees only specific
        # (disentangled) representations, never labels or shared features.
        self.adaptive_main = getattr(args, 'adaptive_main', False)
        if self.adaptive_main:
            self.coalition_aware = getattr(args, 'coalition_aware', False)
            self.contribution_gate = getattr(args, 'contribution_gate', False)
            self.contribution_temperature = getattr(
                args, 'contribution_temperature', 0.2)
            self.adaptive_shared_concat = getattr(args, 'adaptive_shared_concat', False)
            self.gate_norms = nn.ModuleList([nn.LayerNorm(self.d_l) for _ in range(3)])
            self.gate_scores = nn.ModuleList([nn.Linear(self.d_l, 1) for _ in range(3)])
            if self.contribution_gate:
                # Predict signed leave-one-specific-stream-out MAE increments
                # from specific representations only. Labels are never inputs.
                self.contribution_head = nn.Sequential(
                    nn.LayerNorm(3 * self.d_l + 3),
                    nn.Linear(3 * self.d_l + 3, 2 * self.d_l),
                    nn.SiLU(),
                    nn.Linear(2 * self.d_l, 3))
            self.gate_cross_context = getattr(args, 'gate_cross_context', False)
            if self.gate_cross_context:
                # Compare the three modality-specific summaries (and their
                # pairwise differences) before assigning sample-wise roles.
                self.gate_cross = nn.Sequential(
                    nn.Linear(6 * self.d_l, 3 * self.d_l), nn.SiLU(),
                    nn.Linear(3 * self.d_l, 3))
                nn.init.zeros_(self.gate_cross[-1].weight)
                nn.init.zeros_(self.gate_cross[-1].bias)
            self.gate_temperature = getattr(args, 'gate_temperature', 1.0)
            self.counterfactual_route = getattr(args, 'counterfactual_route', False)
            self.counterfactual_specific_only = getattr(
                args, 'counterfactual_specific_only', False)
            self.route_alpha = getattr(args, 'route_alpha', 1.0)
            self.route_temperature = getattr(args, 'route_temperature', 0.1)
            self.dynamic_attention = nn.MultiheadAttention(
                self.d_l, self.num_heads, dropout=self.attn_dropout, batch_first=True)
            fusion_dim = (5 if self.adaptive_shared_concat else 3) * self.d_l
            self.adaptive_proj1 = nn.Linear(fusion_dim, fusion_dim)
            self.adaptive_proj2 = nn.Linear(fusion_dim, fusion_dim)
            self.adaptive_out = nn.Linear(fusion_dim, output_dim)
            # These fixed text-led fusion branches are never executed by the
            # adaptive path; remove them to avoid carrying dead parameters.
            unused = [
                'proj_cosine_l', 'proj_cosine_v', 'proj_cosine_a',
                'self_attentions_c_l', 'self_attentions_c_v', 'self_attentions_c_a',
                'trans_l_with_a', 'trans_l_with_v', 'trans_a_with_l',
                'trans_a_with_v', 'trans_v_with_l', 'trans_v_with_a',
                'trans_l_mem', 'trans_a_mem', 'trans_v_mem',
                'proj1_l_low', 'proj2_l_low', 'out_layer_l_low',
                'proj1_v_low', 'proj2_v_low', 'out_layer_v_low',
                'proj1_a_low', 'proj2_a_low', 'out_layer_a_low',
                'projector_l', 'projector_v', 'projector_a', 'projector_c',
                'proj1', 'proj2', 'out_layer',
            ]
            for name in unused:
                delattr(self, name)
        # Validation-selected decision thresholds are inference metadata, not
        # trainable parameters and therefore do not alter checkpoint keys.
        self.lower = None
        self.upper = None

    @classmethod
    def from_final_checkpoint(cls, root=None, checkpoint_path=None,
                              calibration_path=None, device=None):
        """Load the final question-3 model directly into the original DLF class."""
        from config import get_config_regression

        root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = (Path(checkpoint_path) if checkpoint_path is not None
                           else root / 'checkpoints/final/model.pth')
        calibration_path = (Path(calibration_path) if calibration_path is not None
                            else root / 'outputs/validation/valid_thresholds.json')
        if not checkpoint_path.is_file():
            raise FileNotFoundError('Final checkpoint missing: ' + str(checkpoint_path))
        with calibration_path.open(encoding='utf-8') as stream:
            calibration = json.load(stream)
        lower = float(calibration['threshold_negative_neutral'])
        upper = float(calibration['threshold_neutral_positive'])
        gate_temperature = float(calibration['gate_temperature'])
        if not lower < upper or gate_temperature <= 0:
            raise ValueError('Invalid validation calibration.')
        args = get_config_regression('DLF', 'mosei')
        args.adaptive_main = True
        args.coalition_aware = True
        args.gate_temperature = gate_temperature
        model = cls(args).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device),
                              strict=True)
        model.lower, model.upper = lower, upper
        return model.eval()

    def _explain_coalitions(self, result):
        """Audit the same forward pass, keeping gate and decision main distinct."""
        if self.lower is None or self.upper is None:
            raise ValueError('Load validation thresholds before requesting explanation.')
        available = torch.stack([result['masks'][name].any(dim=1)
                                 for name in ('l', 'a', 'v')], dim=1)
        if not torch.allclose(result['coalition_logits'][:, 7:8],
                              result['output_logit'], atol=1e-5, rtol=1e-5):
            raise AssertionError('Full coalition does not reproduce prediction.')
        if not torch.allclose(result['fixed_coalition_logits'][:, 7:8],
                              result['output_logit'], atol=1e-5, rtol=1e-5):
            raise AssertionError('Fixed-route full coalition differs from prediction.')
        attribution = attribute_coalitions(
            result['coalition_logits'], result['fixed_coalition_logits'],
            available, self.lower, self.upper)
        result['fusion_main_modality'] = result['main_modality']
        result.update(attribution)
        result['main_modality'] = attribution['decision_main_modality']
        return result
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    @staticmethod
    def _project_mask(mask, kernel_size, projected):
        if mask is None:
            return None
        if projected.size(2) == mask.size(1):
            return mask
        return F.max_pool1d(mask.float().unsqueeze(1), kernel_size,
                            stride=1).squeeze(1).bool()

    @staticmethod
    def _last_valid(sequence, mask):
        if mask is None:
            return sequence[-1]
        positions = torch.arange(sequence.size(0), device=sequence.device)
        last = (mask.long() * positions.unsqueeze(0)).max(dim=1).values
        gathered = sequence[last, torch.arange(sequence.size(1), device=sequence.device)]
        return gathered * mask.any(dim=1, keepdim=True).to(gathered.dtype)

    @staticmethod
    def _masked_mean(sequence, mask):
        # sequence is time x batch x hidden
        sequence = sequence.transpose(0, 1)
        if mask is None:
            return sequence.mean(dim=1)
        weights = mask.unsqueeze(-1).to(sequence.dtype)
        return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def _adaptive_fusion(self, s_l, s_a, s_v, c_l, c_a, c_v,
                         mask_l, mask_a, mask_v, return_attention=False,
                         return_counterfactual=False, route_effects=None,
                         counterfactual_train_grad=False,
                         train_coalition=False, return_coalitions=False):
        specific = [s_l, s_a, s_v]
        shared = [c_l, c_a, c_v]
        masks = [mask_l, mask_a, mask_v]
        summaries = torch.stack([self._masked_mean(x, m)
                                 for x, m in zip(specific, masks)], dim=1)
        shared_summaries = torch.stack([self._masked_mean(x, m)
                                        for x, m in zip(shared, masks)], dim=1)
        availability = torch.stack([
            m.any(dim=1) if m is not None else
            torch.ones(x.size(1), dtype=torch.bool, device=x.device)
            for x, m in zip(specific, masks)], dim=1)
        if (train_coalition or return_coalitions) and not availability.any(dim=1).all():
            raise ValueError('Coalition evaluation needs an observed modality.')
        normalized = [norm(summaries[:, i])
                      for i, norm in enumerate(self.gate_norms)]
        gate_logits = torch.cat([
            scorer(normalized[i]) for i, scorer in enumerate(self.gate_scores)
        ], dim=1)
        if self.gate_cross_context:
            x_l, x_a, x_v = normalized
            cross_features = torch.cat(
                [x_l, x_a, x_v, x_l - x_a, x_l - x_v, x_a - x_v], dim=1)
            gate_logits = gate_logits + self.gate_cross(cross_features)
        gate_logits = gate_logits.masked_fill(~availability, -1e4)
        predicted_gains = None
        if self.contribution_gate:
            if self.contribution_temperature <= 0:
                raise ValueError('Contribution temperature must be positive.')
            predicted_gains = self.contribution_head(torch.cat(
                [summaries.flatten(start_dim=1), availability.float()], dim=1))
            route_logits = predicted_gains.masked_fill(~availability, -1e4)
            route_temperature = self.contribution_temperature
        else:
            route_logits = gate_logits
            route_temperature = self.gate_temperature
        weights = F.softmax(route_logits / route_temperature, dim=1)
        def fuse(role_weights, role_masks, attention_requested=False,
                 deterministic=False, shared_masks=None, shared_role_weights=None):
            if shared_masks is None:
                shared_masks = role_masks
            if shared_role_weights is None:
                shared_role_weights = role_weights
            query = (role_weights.unsqueeze(-1) * summaries).sum(dim=1, keepdim=True)
            # One attention operation consumes all specific and shared streams.
            tokens = torch.cat([
                x.transpose(0, 1) * role_weights[:, i, None, None]
                for i, x in enumerate(specific)
            ] + [x.transpose(0, 1) for x in shared], dim=1)
            valid = torch.cat([
                m if m is not None else torch.ones(x.size(1), x.size(0),
                                                   dtype=torch.bool, device=x.device)
                for x, m in zip(specific + shared, role_masks + shared_masks)
            ], dim=1)
            context, attention = self.dynamic_attention(
                query, tokens, tokens, key_padding_mask=~valid,
                need_weights=attention_requested, average_attn_weights=False)
            if self.adaptive_shared_concat:
                shared_summary = (shared_summaries *
                                  torch.stack([m.any(dim=1) if m is not None else
                                               torch.ones(x.size(1), dtype=torch.bool,
                                                          device=x.device)
                                               for x, m in zip(shared, shared_masks)], dim=1)
                                  .unsqueeze(-1)).flatten(start_dim=1)
            else:
                shared_summary = (shared_role_weights.unsqueeze(-1) *
                                  shared_summaries).sum(dim=1)
            fused = torch.cat([query.squeeze(1), context.squeeze(1), shared_summary], dim=-1)
            projected = self.adaptive_proj2(F.dropout(
                F.relu(self.adaptive_proj1(fused)), p=self.output_dropout,
                training=self.training and not deterministic)) + fused
            return self.adaptive_out(projected), attention

        output, attention = fuse(weights, masks, return_attention)
        ablations = None
        reference = None
        if self.counterfactual_route or (return_counterfactual and route_effects is None):
            # Re-fuse the *same encoded representations* with one modality
            # absent. This avoids rerunning BERT on out-of-distribution zeros.
            was_training = self.dynamic_attention.training
            self.dynamic_attention.eval()
            try:
                with torch.no_grad():
                    reference, _ = fuse(weights.detach(), masks, deterministic=True)
                    variants = []
                    for i in range(3):
                        remaining = availability.clone()
                        remaining[:, i] = False
                        has_other = remaining.any(dim=1)
                        remaining = torch.where(has_other.unsqueeze(1), remaining, availability)
                        omitted_weights = F.softmax(
                            (route_logits.detach() / route_temperature).masked_fill(
                                ~remaining, -1e4), dim=1)
                        omitted_masks = [
                            (m if m is not None else torch.ones(x.size(1), x.size(0),
                                                               dtype=torch.bool, device=x.device)) &
                            remaining[:, j, None]
                            for j, (x, m) in enumerate(zip(specific, masks))
                        ]
                        changed, _ = fuse(
                            omitted_weights, omitted_masks, deterministic=True,
                            shared_masks=masks if self.counterfactual_specific_only else None,
                            shared_role_weights=weights.detach()
                            if self.counterfactual_specific_only else None)
                        variants.append(changed)
                    ablations = torch.cat(variants, dim=1)
            finally:
                self.dynamic_attention.train(was_training)
        if self.counterfactual_route or route_effects is not None:
            if not 0 <= self.route_alpha <= 1 or self.route_temperature <= 0:
                raise ValueError('Counterfactual route alpha/temperature is invalid.')
            effects = ((ablations - reference).abs() if route_effects is None
                       else route_effects.to(weights.device))
            effects = effects.masked_fill(~availability, -1e4)
            counterfactual_weights = F.softmax(
                effects / self.route_temperature, dim=1)
            weights = ((1 - self.route_alpha) * weights +
                       self.route_alpha * counterfactual_weights)
            output, attention = fuse(weights, masks, return_attention)
        if return_counterfactual and route_effects is not None:
            # Diagnose the *final routed predictor*, not the preliminary
            # reference gate. Keep the external route fixed under omission.
            was_training = self.dynamic_attention.training
            self.dynamic_attention.eval()
            try:
                with (nullcontext() if counterfactual_train_grad else torch.no_grad()):
                    fixed_weights = weights.detach()
                    reference, _ = fuse(fixed_weights, masks,
                                        deterministic=True)
                    variants = []
                    for i in range(3):
                        remaining = availability.clone()
                        remaining[:, i] = False
                        has_other = remaining.any(dim=1)
                        remaining = torch.where(has_other.unsqueeze(1), remaining,
                                                availability)
                        omitted_weights = fixed_weights.masked_fill(~remaining, 0)
                        omitted_weights = (omitted_weights /
                                           omitted_weights.sum(dim=1, keepdim=True).clamp_min(1e-8))
                        omitted_masks = [
                            (m if m is not None else torch.ones(x.size(1), x.size(0),
                                                               dtype=torch.bool, device=x.device)) &
                            remaining[:, j, None]
                            for j, (x, m) in enumerate(zip(specific, masks))
                        ]
                        changed, _ = fuse(
                            omitted_weights, omitted_masks, deterministic=True,
                            shared_masks=masks if self.counterfactual_specific_only else None,
                            shared_role_weights=fixed_weights
                            if self.counterfactual_specific_only else None)
                        variants.append(changed)
                    ablations = torch.cat(variants, dim=1)
            finally:
                self.dynamic_attention.train(was_training)
        coalition_train_logit = None
        if train_coalition:
            if not self.coalition_aware:
                raise ValueError('Coalition training requires coalition_aware=True.')
            # Uniformly sample one proper nonempty coalition for each example.
            # This trains the SAME fusion head used by the full prediction.
            random_bits = torch.randint(1, 7, (availability.size(0),),
                                        device=availability.device)
            selected = torch.stack([(random_bits & (1 << i)) != 0
                                    for i in range(3)], dim=1) & availability
            fallback = availability.long().argmax(dim=1)
            selected = selected | (
                ~selected.any(dim=1, keepdim=True) &
                F.one_hot(fallback, 3).bool())
            subset_weights = weights.masked_fill(~selected, 0)
            subset_weights = subset_weights / subset_weights.sum(
                dim=1, keepdim=True).clamp_min(1e-8)
            subset_masks = [m & selected[:, i, None]
                            for i, m in enumerate(masks)]
            coalition_train_logit, _ = fuse(subset_weights, subset_masks)
        coalition_logits = fixed_coalition_logits = None
        if return_coalitions:
            if not self.coalition_aware:
                raise ValueError('Coalition inference requires coalition_aware=True.')
            if self.training:
                raise ValueError('Coalition inference requires eval mode.')
            rerouted, controlled = [], []
            for bits in range(8):
                selected = availability & torch.tensor(
                    [bool(bits & (1 << i)) for i in range(3)],
                    device=availability.device)
                has_any = selected.any(dim=1)
                # The no-evidence reference score is the neutral intensity 0.
                # Dummy valid keys prevent all-masked attention NaNs; results
                # for empty rows are replaced immediately by the baseline.
                safe = torch.where(has_any[:, None], selected,
                                   F.one_hot(availability.long().argmax(dim=1),
                                             3).bool())
                subset_masks = [m & safe[:, i, None]
                                for i, m in enumerate(masks)]
                gated = weights.masked_fill(~safe, 0)
                gated = gated / gated.sum(dim=1, keepdim=True).clamp_min(1e-8)
                fixed = weights.masked_fill(~safe, 0)
                changed, _ = fuse(gated, subset_masks, deterministic=True)
                controlled_changed, _ = fuse(fixed, subset_masks,
                                             deterministic=True)
                rerouted.append(torch.where(has_any[:, None], changed,
                                            torch.zeros_like(changed)))
                controlled.append(torch.where(has_any[:, None],
                                              controlled_changed,
                                              torch.zeros_like(controlled_changed)))
            coalition_logits = torch.cat(rerouted, dim=1)
            fixed_coalition_logits = torch.cat(controlled, dim=1)
        return (output, weights, attention, reference, ablations,
                predicted_gains, coalition_train_logit, coalition_logits,
                fixed_coalition_logits)

    def forward(self, text, audio, video, text_mask=None, audio_mask=None,
                vision_mask=None, return_attention=False,
                return_counterfactual=False, route_effects=None,
                counterfactual_train_grad=False, train_coalition=False,
                return_coalitions=False, return_explanation=False):
        if return_explanation:
            if not self.adaptive_main or not self.coalition_aware:
                raise ValueError('Explanation requires coalition-aware adaptive DLF.')
            return_coalitions = True
        #extraction
        if text_mask is None and self.use_bert:
            text_mask = text[:, 1, :].bool()
        if self.use_bert:
            text = self.text_model(text)
        if text_mask is not None:
            text = text * text_mask.unsqueeze(-1).to(text.dtype)
        if audio_mask is not None:
            audio = audio * audio_mask.unsqueeze(-1).to(audio.dtype)
        if vision_mask is not None:
            video = video * vision_mask.unsqueeze(-1).to(video.dtype)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training) 
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) 
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a) 
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        mask_l = self._project_mask(text_mask, self.kernel_l, proj_x_l)
        mask_a = self._project_mask(audio_mask, self.kernel_a, proj_x_a)
        mask_v = self._project_mask(vision_mask, self.kernel_v, proj_x_v)
        if mask_l is not None:
            proj_x_l = proj_x_l * mask_l.unsqueeze(1).to(proj_x_l.dtype)
        if mask_a is not None:
            proj_x_a = proj_x_a * mask_a.unsqueeze(1).to(proj_x_a.dtype)
        if mask_v is not None:
            proj_x_v = proj_x_v * mask_v.unsqueeze(1).to(proj_x_v.dtype)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)   
        proj_x_v = proj_x_v .permute(2, 0, 1)  
        proj_x_a = proj_x_a.permute(2, 0, 1)

        #disentanglement
        s_l = self.encoder_s_l(proj_x_l, query_mask=mask_l)
        s_v = self.encoder_s_v(proj_x_v, query_mask=mask_v)
        s_a = self.encoder_s_a(proj_x_a, query_mask=mask_a)

        c_l = self.encoder_c(proj_x_l, query_mask=mask_l)
        c_v = self.encoder_c(proj_x_v, query_mask=mask_v)
        c_a = self.encoder_c(proj_x_a, query_mask=mask_a)


        s_l = s_l.permute(1, 2, 0)   
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]


        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l, query_mask=mask_l).permute(1, 2, 0)
        s_v_r = self.encoder_s_v(recon_v, query_mask=mask_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a, query_mask=mask_a).permute(1, 2, 0)
        
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        if self.adaptive_main:
            (output, role_weights, attention, reference_logits,
             ablation_logits, predicted_gains, coalition_train_logit,
             coalition_logits, fixed_coalition_logits) = self._adaptive_fusion(
                s_l, s_a, s_v, c_l, c_a, c_v,
                mask_l, mask_a, mask_v, return_attention,
                return_counterfactual, route_effects,
                counterfactual_train_grad, train_coalition,
                return_coalitions)
            # Balanced auxiliary heads supervise each available specific
            # modality without making text the privileged prediction path.
            specific_pools = [self._masked_mean(s, m) for s, m in
                              [(s_l, mask_l), (s_v, mask_v), (s_a, mask_a)]]
            high_logits = []
            for pool, proj1, proj2, head in zip(
                    specific_pools,
                    [self.proj1_l_high, self.proj1_v_high, self.proj1_a_high],
                    [self.proj2_l_high, self.proj2_v_high, self.proj2_a_high],
                    [self.out_layer_l_high, self.out_layer_v_high, self.out_layer_a_high]):
                hidden = proj2(F.dropout(F.relu(proj1(pool)),
                                         p=self.output_dropout, training=self.training)) + pool
                high_logits.append(head(hidden))
            shared_pools = [self._masked_mean(c, m) for c, m in
                            [(c_l, mask_l), (c_v, mask_v), (c_a, mask_a)]]
            shared_cat = torch.cat(shared_pools, dim=1)
            shared_hidden = self.proj2_c(F.dropout(F.relu(self.proj1_c(shared_cat)),
                                                   p=self.output_dropout,
                                                   training=self.training)) + shared_cat
            result = {
                'origin_l': proj_x_l, 'origin_v': proj_x_v, 'origin_a': proj_x_a,
                's_l': s_l, 's_v': s_v, 's_a': s_a,
                'c_l': c_l, 'c_v': c_v, 'c_a': c_a,
                's_l_r': s_l_r, 's_v_r': s_v_r, 's_a_r': s_a_r,
                'recon_l': recon_l, 'recon_v': recon_v, 'recon_a': recon_a,
                'c_l_sim': c_l_sim, 'c_v_sim': c_v_sim, 'c_a_sim': c_a_sim,
                'logits_l_hetero': high_logits[0],
                'logits_v_hetero': high_logits[1],
                'logits_a_hetero': high_logits[2],
                'logits_c': self.out_layer_c(shared_hidden),
                'output_logit': output,
                'role_weights': role_weights,
                'predicted_gains': predicted_gains,
                'main_modality': role_weights.argmax(dim=1),
                'attention': attention,
                'reference_logits': reference_logits,
                'ablation_logits': ablation_logits,
                'coalition_train_logit': coalition_train_logit,
                'coalition_logits': coalition_logits,
                'fixed_coalition_logits': fixed_coalition_logits,
                'masks': {'l': mask_l, 'a': mask_a, 'v': mask_v},
            }
            return self._explain_coalitions(result) if return_explanation else result
       
       #enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)  
        repr_l_low = self.proj1_l_low(hs_l_low)                            
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low         
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        
        c_l_att = self.self_attentions_c_l(c_l, query_mask=mask_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = self._last_valid(c_l_att, mask_l)

        c_v_att = self.self_attentions_c_v(c_v, query_mask=mask_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = self._last_valid(c_v_att, mask_v)

        c_a_att = self.self_attentions_c_a(c_a, query_mask=mask_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = self._last_valid(c_a_att, mask_a)

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion                       
        logits_c = self.out_layer_c(c_proj)     
        
        # LFA
        # L --> L                
        h_ls = s_l                     
        h_ls = self.trans_l_mem(h_ls, query_mask=mask_l)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = self._last_valid(h_ls, mask_l)

        # A --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a,
                                         query_mask=mask_l, key_mask=mask_a)
        h_as = h_l_with_as
        h_as = self.trans_a_mem(h_as, query_mask=mask_l)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = self._last_valid(h_as, mask_l)
        if mask_a is not None:
            last_h_a = last_h_a * mask_a.any(dim=1, keepdim=True).to(last_h_a.dtype)

        # V --> L
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v,
                                         query_mask=mask_l, key_mask=mask_v)
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs, query_mask=mask_l)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = self._last_valid(h_vs, mask_l)
        if mask_v is not None:
            last_h_v = last_h_v * mask_v.any(dim=1, keepdim=True).to(last_h_v.dtype)


        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)
        
        #fusion
        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))   
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        if mask_v is not None:
            last_h_v = last_h_v * mask_v.any(dim=1, keepdim=True).to(last_h_v.dtype)
        if mask_a is not None:
            last_h_a = last_h_a * mask_a.any(dim=1, keepdim=True).to(last_h_a.dtype)
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)   

        #prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs                          

        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,        
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,           
            'logits_v_hetero': logits_v_high, 
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,
            'output_logit': output
        }
        res['masks'] = {'l': mask_l, 'a': mask_a, 'v': mask_v}
        return res
