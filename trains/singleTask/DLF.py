import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss


logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real, mask=None):
        diffs = torch.add(real, -pred)
        if mask is None:
            return torch.mean(diffs.pow(2))
        # pred/real are time x batch x channel.
        weights = mask.T.unsqueeze(-1).to(diffs.dtype)
        return torch.sum(diffs.pow(2) * weights) / (weights.sum() * diffs.size(-1)).clamp_min(1)

class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    @staticmethod
    def _batch_masks(batch_data, device):
        return {
            name: batch_data[key].to(device) if key in batch_data else None
            for name, key in [('text_mask', 'text_mask'),
                              ('audio_mask', 'audio_mask'),
                              ('vision_mask', 'vision_mask')]
        }

    def _masked_aux_loss(self, pred, real, mask):
        if mask is None:
            return self.criterion(pred, real)
        available = mask.any(dim=1)
        if not available.any():
            return pred.sum() * 0
        return self.criterion(pred[available], real[available])

    def _orthogonality_loss(self, specific, shared, mask):
        if mask is not None:
            specific = specific.transpose(0, 1)[mask]
            shared = shared.transpose(0, 1)[mask]
        else:
            specific = specific.reshape(-1, specific.size(-1))
            shared = shared.reshape(-1, shared.size(-1))
        if specific.numel() == 0:
            return specific.sum() * 0
        target = torch.full((specific.size(0),), -1, device=specific.device)
        return self.cosine(specific, shared, target)

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        teacher_gains = None
        if getattr(self.args, 'gate_teacher_targets', None):
            cached = np.load(self.args.gate_teacher_targets)
            teacher_gains = cached['gains']
            if teacher_gains.shape != (len(dataloader['train'].dataset), 3):
                raise ValueError('Fixed gate-teacher gains do not match training split.')
        route_train_effects = None
        if getattr(self.args, 'route_teacher_train_targets', None):
            route_train_effects = np.load(self.args.route_teacher_train_targets)['effects']
            if route_train_effects.shape != (len(dataloader['train'].dataset), 3):
                raise ValueError('Fixed route effects do not match training split.')
        contribution_targets = None
        if getattr(self.args, 'contribution_train_targets', None):
            cached = np.load(self.args.contribution_train_targets)
            contribution_targets = (cached['gains'], cached['available'])
            if contribution_targets[0].shape != (
                    len(dataloader['train'].dataset), 3):
                raise ValueError('Contribution targets do not match training split.')
            if not np.isfinite(contribution_targets[0]).all():
                raise ValueError('Contribution targets are incomplete or non-finite.')
        if getattr(self.args, 'train_contribution_head_only', False):
            if contribution_targets is None or not getattr(model[0], 'contribution_gate', False):
                raise ValueError('Head-only training requires contribution targets and gate.')
            for param in model[0].parameters():
                param.requires_grad_(False)
            head_params = list(model[0].contribution_head.parameters())
            for param in head_params:
                param.requires_grad_(True)
            optimizer = optim.Adam(head_params, lr=self.args.learning_rate)
        elif getattr(self.args, 'train_fusion_only', False):
            for param in model[0].parameters():
                param.requires_grad_(False)
            fusion_params = (list(model[0].dynamic_attention.parameters()) +
                             list(model[0].adaptive_proj1.parameters()) +
                             list(model[0].adaptive_proj2.parameters()) +
                             list(model[0].adaptive_out.parameters()))
            for param in fusion_params:
                param.requires_grad_(True)
            optimizer = optim.Adam(fusion_params, lr=self.args.learning_rate)
        elif getattr(self.args, 'train_gate_only', False):
            for param in model[0].parameters():
                param.requires_grad_(False)
            gate_params = list(model[0].gate_norms.parameters()) + list(model[0].gate_scores.parameters())
            if getattr(model[0], 'gate_cross_context', False):
                gate_params += list(model[0].gate_cross.parameters())
            for param in gate_params:
                param.requires_grad_(True)
            optimizer = optim.Adam(gate_params,
                                   lr=getattr(self.args, 'gate_learning_rate', 1e-3))
        elif getattr(self.args, 'init_checkpoint', None) and model[0].use_bert:
            bert_params = list(model[0].text_model.parameters())
            bert_ids = {id(p) for p in bert_params}
            gate_lr = getattr(self.args, 'gate_learning_rate', None)
            if gate_lr is not None:
                gate_params = list(model[0].gate_norms.parameters()) + list(model[0].gate_scores.parameters())
                if getattr(model[0], 'gate_cross_context', False):
                    gate_params += list(model[0].gate_cross.parameters())
                gate_ids = {id(p) for p in gate_params}
                other_params = [p for p in model[0].parameters()
                                if id(p) not in bert_ids and id(p) not in gate_ids]
                optimizer = optim.Adam([
                    {'params': bert_params, 'lr': getattr(self.args, 'backbone_learning_rate', 1e-5)},
                    {'params': other_params, 'lr': self.args.learning_rate},
                    {'params': gate_params, 'lr': gate_lr},
                ])
            else:
                new_params = [p for p in model[0].parameters() if id(p) not in bert_ids]
                optimizer = optim.Adam([
                    {'params': bert_params,
                     'lr': getattr(self.args, 'backbone_learning_rate', 1e-5)},
                    {'params': new_params, 'lr': self.args.learning_rate},
                ])
        else:
            optimizer = optim.Adam(model[0].parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'GainMAE', 'MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)    
        model = net
        
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()
            if getattr(self.args, 'train_contribution_head_only', False):
                # Frozen encoder must produce deterministic teacher features.
                model[0].eval()
                model[0].contribution_head.train()
              

            train_loss = 0.0
            accumulated = 0
            optimizer.zero_grad()
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    masks = self._batch_masks(batch_data, self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)


                   
                    use_counterfactual = (getattr(self.args, 'gate_counterfactual_weight', 0.0) > 0
                                          and teacher_gains is None)
                    consistency_weight = getattr(
                        self.args, 'contribution_consistency_weight', 0.0)
                    use_counterfactual = use_counterfactual or consistency_weight > 0
                    route_effects = (torch.as_tensor(
                        route_train_effects[batch_data['index'].numpy()],
                        dtype=labels.dtype, device=labels.device)
                        if route_train_effects is not None else None)
                    output = model[0](text, audio, vision, **masks,
                                      return_counterfactual=use_counterfactual,
                                      route_effects=route_effects,
                                      counterfactual_train_grad=consistency_weight > 0,
                                      train_coalition=(getattr(
                                          self.args, 'coalition_loss_weight', 0.0) > 0))

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    
                    loss_task_l_hetero = self._masked_aux_loss(output['logits_l_hetero'], labels, output['masks']['l'])
                    loss_task_v_hetero = self._masked_aux_loss(output['logits_v_hetero'], labels, output['masks']['v'])
                    loss_task_a_hetero = self._masked_aux_loss(output['logits_a_hetero'], labels, output['masks']['a'])
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    
                    # total MSA loss L_msa
                    if getattr(self.args, 'adaptive_main', False):
                        loss_task = (loss_task_all + 0.3 * loss_task_c +
                                     0.2 * (loss_task_l_hetero + loss_task_v_hetero +
                                            loss_task_a_hetero))
                    else:
                        loss_task = (loss_task_all + loss_task_c +
                                     3 * loss_task_l_hetero + loss_task_v_hetero +
                                     loss_task_a_hetero)
                    
                    # reconstruction loss L_r
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'], output['masks']['l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'], output['masks']['v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'], output['masks']['a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s 
                    loss_sl_slr = self.MSE(output['s_l'], output['s_l_r'].permute(2, 0, 1), output['masks']['l'])
                    loss_sv_slv = self.MSE(output['s_v'], output['s_v_r'].permute(2, 0, 1), output['masks']['v'])
                    loss_sa_sla = self.MSE(output['s_a'], output['s_a_r'].permute(2, 0, 1), output['masks']['a'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    cosine_similarity_s_c_l = self._orthogonality_loss(output['s_l'], output['c_l'], output['masks']['l'])
                    cosine_similarity_s_c_v = self._orthogonality_loss(output['s_v'], output['c_v'], output['masks']['v'])
                    cosine_similarity_s_c_a = self._orthogonality_loss(output['s_a'], output['c_a'], output['masks']['a'])
                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # triplet margin loss L_m
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        for name, feature in [('l', c_l), ('v', c_v), ('a', c_a)]:
                            mask = output['masks'][name]
                            if mask is None or mask[i].any():
                                feats.append(feature[i].view(1, -1))
                                ids.append(labels[i].view(1, -1))
                    if len(feats) > 1:
                        loss_sim = self.sim_loss(torch.cat(ids), torch.cat(feats))
                    else:
                        loss_sim = output['output_logit'].sum() * 0

                    #overall loss L_DLF
                    combined_loss = loss_task + (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1   
                    if getattr(self.args, 'coalition_loss_weight', 0.0) > 0:
                        combined_loss = (
                            combined_loss + self.args.coalition_loss_weight *
                            self.criterion(output['coalition_train_logit'], labels))

                    if (getattr(self.args, 'adaptive_main', False) and
                            not getattr(self.args, 'contribution_gate', False)):
                        # Train-time labels supervise which specific modality
                        # is locally reliable. At inference the gate receives
                        # only specific representations, never labels.
                        unimodal = torch.cat([
                            output['logits_l_hetero'],
                            output['logits_a_hetero'],
                            output['logits_v_hetero']], dim=1)
                        available = torch.stack([
                            output['masks'][name].any(dim=1)
                            for name in ('l', 'a', 'v')], dim=1)
                        reliability_temp = getattr(self.args, 'gate_reliability_temperature', 0.5)
                        reliability = -(unimodal.detach() - labels).abs() / reliability_temp
                        reliability = reliability.masked_fill(~available, -1e4)
                        target_roles = torch.softmax(reliability, dim=1)
                        role_weights = output['role_weights'].clamp_min(1e-8)
                        gate_supervision = -(target_roles * role_weights.log()).sum(dim=1).mean()
                        available_prior = (available.float() /
                                           available.sum(dim=1, keepdim=True).clamp_min(1)).mean(dim=0)
                        gate_balance = (role_weights.mean(dim=0) - available_prior).square().sum()
                        gate_entropy = -(role_weights * role_weights.log()).sum(dim=1).mean()
                        combined_loss = (
                            combined_loss +
                            getattr(self.args, 'gate_supervision_weight', 0.25) * gate_supervision +
                            getattr(self.args, 'gate_balance_weight', 2.0) * gate_balance +
                            getattr(self.args, 'gate_entropy_weight', 0.03) * gate_entropy)
                        if getattr(self.args, 'gate_counterfactual_weight', 0.0) > 0:
                            # A modality is useful if removing it increases
                            # label error. Supervise only clear, positive
                            # cases; ambiguous cases retain the task loss.
                            if teacher_gains is None:
                                errors_without = (output['ablation_logits'].detach() - labels).abs()
                                error_full = (output['reference_logits'] - labels).abs()
                                gains = errors_without - error_full
                            else:
                                gains = torch.as_tensor(
                                    teacher_gains[batch_data['index'].numpy()],
                                    dtype=labels.dtype, device=labels.device)
                            gains = gains.masked_fill(~available, -1e4)
                            ranked = gains.topk(2, dim=1).values
                            clear = ((available.sum(dim=1) > 1) &
                                     (ranked[:, 0] >= getattr(self.args, 'gate_cf_min_gain', 0.02)) &
                                     ((ranked[:, 0] - ranked[:, 1]) >=
                                      getattr(self.args, 'gate_cf_min_margin', 0.02)))
                            if clear.any():
                                temperature = getattr(self.args, 'gate_cf_target_temperature', 0.1)
                                targets = torch.softmax(gains[clear] / temperature, dim=1)
                                cf_loss = -(targets * role_weights[clear].log()).sum(dim=1).mean()
                                combined_loss = (combined_loss +
                                    getattr(self.args, 'gate_counterfactual_weight', 0.0) * cf_loss)

                    if contribution_targets is not None:
                        target_gains, target_available = contribution_targets
                        gains = torch.as_tensor(
                            target_gains[batch_data['index'].numpy()],
                            dtype=labels.dtype, device=labels.device)
                        target_mask = torch.as_tensor(
                            target_available[batch_data['index'].numpy()],
                            dtype=torch.bool, device=labels.device)
                        predicted = output['predicted_gains']
                        if predicted is None:
                            raise ValueError('Contribution targets require a contribution gate.')
                        gain_loss = torch.nn.functional.smooth_l1_loss(
                            predicted[target_mask], gains[target_mask])
                        if getattr(self.args, 'train_contribution_head_only', False):
                            combined_loss = gain_loss
                        else:
                            combined_loss = (combined_loss +
                                getattr(self.args, 'contribution_gain_loss_weight', 0.0)
                                * gain_loss)

                    if consistency_weight > 0:
                        if route_effects is None:
                            raise ValueError('Contribution consistency requires fixed route targets.')
                        available = torch.stack([
                            output['masks'][name].any(dim=1)
                            for name in ('l', 'a', 'v')], dim=1)
                        actual_gains = ((output['ablation_logits'] - labels).abs() -
                                        (output['reference_logits'] - labels).abs())
                        target_gains = route_effects.detach()
                        consistency_loss = torch.nn.functional.smooth_l1_loss(
                            actual_gains[available], target_gains[available])
                        selected = target_gains.masked_fill(~available, -1e4).argmax(dim=1)
                        selected_actual = actual_gains.gather(1, selected[:, None])
                        helpful_loss = torch.relu(0.02 - selected_actual).mean()
                        combined_loss = (combined_loss + consistency_weight *
                                         (consistency_loss + 0.2 * helpful_loss))

                    text_drop_probability = getattr(
                        self.args, 'route_text_drop_probability', 0.0)
                    if text_drop_probability > 0:
                        # Train the adapted fusion head to retain useful
                        # audio/vision evidence when text is unavailable.
                        other_available = (masks['audio_mask'].any(dim=1) |
                                           masks['vision_mask'].any(dim=1))
                        drop = ((torch.rand(labels.size(0), device=labels.device)
                                 < text_drop_probability) & other_available)
                        if drop.any():
                            dropout_masks = dict(masks)
                            dropout_masks['text_mask'] = masks['text_mask'].clone()
                            dropout_masks['text_mask'][drop] = False
                            dropout_output = model[0](
                                text, audio, vision, **dropout_masks,
                                route_effects=route_effects)
                            dropout_loss = self.criterion(
                                dropout_output['output_logit'][drop], labels[drop])
                            combined_loss = (
                                combined_loss +
                                getattr(self.args, 'route_text_drop_weight', 0.2)
                                * dropout_loss)
                
                    combined_loss.backward()
                    accumulated += 1

                    train_loss += combined_loss.item()
                    

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if accumulated == self.args.update_epochs:
                        if self.args.grad_clip != -1.0:
                            nn.utils.clip_grad_value_(model[0].parameters(), self.args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        accumulated = 0
                if accumulated:
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(model[0].parameters(), self.args.grad_clip)
                    optimizer.step()
            

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results[self.args.KeyEval])
            # Keep only the checkpoint selected by the validation metric.
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model[0].state_dict(), self.args.model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
            if epochs >= getattr(self.args, 'max_epochs', 1000000):
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        eval_count = 0
        route_valid_effects = None
        if getattr(self.args, 'route_teacher_valid_targets', None):
            route_valid_effects = np.load(self.args.route_teacher_valid_targets)['effects']
        contribution_valid = None
        if getattr(self.args, 'contribution_valid_targets', None):
            cached = np.load(self.args.contribution_valid_targets)
            contribution_valid = (cached['gains'], cached['available'])
        gain_abs_error = 0.0
        gain_count = 0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    masks = self._batch_masks(batch_data, self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    route_effects = (torch.as_tensor(
                        route_valid_effects[batch_data['index'].numpy()],
                        dtype=labels.dtype, device=labels.device)
                        if route_valid_effects is not None else None)
                    output = model(text, audio, vision, **masks,
                                   route_effects=route_effects)
                    if contribution_valid is not None:
                        target_gains, target_available = contribution_valid
                        indices = batch_data['index'].numpy()
                        gains = torch.as_tensor(target_gains[indices],
                                                dtype=labels.dtype, device=labels.device)
                        gain_mask = torch.as_tensor(target_available[indices],
                                                    dtype=torch.bool, device=labels.device)
                        predicted = output['predicted_gains']
                        if predicted is None:
                            raise ValueError('Contribution validation requires a contribution gate.')
                        gain_abs_error += (predicted[gain_mask] - gains[gain_mask]).abs().sum().item()
                        gain_count += int(gain_mask.sum().item())
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item() * labels.size(0)
                    eval_count += labels.size(0)
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / eval_count
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = eval_loss
        if contribution_valid is not None:
            eval_results['GainMAE'] = gain_abs_error / max(gain_count, 1)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
