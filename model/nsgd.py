"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from .model import UniterPreTrainedModel, UniterModel
from .layer import GELU, BertOnlyMLMHead


def repeat_interleave(x, n_repeat=1, dim=0):
    repeat_list = [1] * (dim + 1) + [n_repeat] + [1] * (x.dim() - dim - 1)
    x_size = list(x.size())
    x_size[dim] = x_size[dim] * n_repeat
    x = x.unsqueeze(dim+1).repeat(*repeat_list).view(*x_size)
    return x


class UniterForNSGD(UniterPreTrainedModel):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2, hard_size=16, nsgd_sample_size=16, nsgd_sample_temperature=1.0):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)
        self.hard_size = hard_size
        self.nsgd_sample_size = nsgd_sample_size
        self.nsgd_sample_temperature = nsgd_sample_temperature

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward_uniter(
            self,
            sample_size=None,
            input_ids=None, position_ids=None,
            img_feat=None, img_pos_feat=None,
            attn_masks=None, gather_index=None,
            compute_loss=True, sigmoid_norm=False
        ):
        model_outputs = {}
        sequence_output = self.uniter(
            input_ids, position_ids,
            img_feat, img_pos_feat,
            attn_masks, gather_index,
            output_all_encoded_layers=False
        )
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)
        model_outputs['rank_scores'] = rank_scores
        if compute_loss:
            # triplet loss
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            # sample_size = batch['sample_size']
            scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
            pos = scores[:, :1]
            neg = scores[:, 1:]
            rank_loss = torch.clamp(self.margin + neg - pos, 0)
            rank_corrects = neg.sub(pos).le(0).float()
            model_outputs['rank_loss'] = rank_loss
            model_outputs['rank_corrects'] = rank_corrects
        if sigmoid_norm:
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            model_outputs['rank_scores_sigmoid'] = rank_scores_sigmoid
            # sample_size = batch['sample_size']
        return model_outputs

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mlm(
            self,
            input_ids=None, position_ids=None,
            img_feat=None, img_pos_feat=None,
            attn_masks=None, gather_index=None,
            txt_labels=None, compute_loss=True, sampling=True
        ):
        model_outputs = {}
        sequence_output = self.uniter(
            input_ids, position_ids,
            img_feat, img_pos_feat,
            attn_masks, gather_index,
            output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]

        if compute_loss:
            # only compute masked tokens for better efficiency
            masked_output = self._compute_masked_hidden(sequence_output, txt_labels != -1)
            prediction_scores = self.cls(masked_output)
            masked_lm_loss = F.cross_entropy(prediction_scores, txt_labels[txt_labels != -1], reduction='none')
            model_outputs['masked_lm_loss'] = masked_lm_loss
            model_outputs['mlm_corrects'] = prediction_scores.max(-1)[1].eq(txt_labels[txt_labels != -1]).float()
        if sampling:
            bsz, caption_len = input_ids.size(0), input_ids.size(1)
            prediction_scores = self.cls(sequence_output)
            sample_caption_tokens = torch.multinomial(
                prediction_scores.div(self.nsgd_sample_temperature).softmax(-1).view(-1, prediction_scores.size(-1)),
                num_samples=self.nsgd_sample_size,
                replacement=True,
            ).view(bsz, caption_len, self.nsgd_sample_size).permute(0, 2, 1)
            mask_indicator = txt_labels.ne(-1).long().unsqueeze(1)
            synthetic_input_ids = input_ids.unsqueeze(1).mul(1-mask_indicator).\
                add(sample_caption_tokens.mul(mask_indicator)).reshape(-1, caption_len)
            model_outputs['fill_input_ids'] = synthetic_input_ids
        return model_outputs

    def forward(self, batch, sample_from='t', compute_loss=True, compute_mlm=False):
        # expect same input_ids for all pairs
        model_outputs = {}
        if not sample_from.startswith('g'):
            batch_size = batch['attn_masks'].size(0)
            input_ids = batch['input_ids']
            img_feat = batch['img_feat']
            img_pos_feat = batch['img_pos_feat']
            if sample_from == 't':
                if input_ids.size(0) == 1:
                    batch['input_ids'] = input_ids.expand(batch_size, -1)
            elif sample_from == 'i':
                if img_feat.size(0) == 1:
                    batch['img_feat'] = img_feat.expand(batch_size, -1, -1)
                if img_pos_feat.size(0) == 1:
                    batch['img_pos_feat'] = img_pos_feat.expand(batch_size, -1, -1)
            else:
                raise ValueError()

            if self.training and compute_loss:
                with torch.no_grad():
                    self.eval()
                    # print(f'| is_training: {self.training} | compute_loss: {compute_loss} |')
                    if torch.isnan(batch['input_ids']).sum().item() > 0 or \
                            torch.isnan(batch['position_ids']).sum().item() > 0 or \
                            torch.isnan(batch['img_feat']).sum().item() > 0 or \
                            torch.isnan(batch['img_pos_feat']).sum().item() > 0 or \
                            torch.isnan(batch['attn_masks']).sum().item() > 0 or \
                            torch.isnan(batch['gather_index']).sum().item() > 0:
                        print(' | nan appear!')
                    if torch.isinf(batch['input_ids']).sum().item() > 0 or \
                            torch.isinf(batch['position_ids']).sum().item() > 0 or \
                            torch.isinf(batch['img_feat']).sum().item() > 0 or \
                            torch.isinf(batch['img_pos_feat']).sum().item() > 0 or \
                            torch.isinf(batch['attn_masks']).sum().item() > 0 or \
                            torch.isinf(batch['gather_index']).sum().item() > 0:
                        print(' | inf appear!')

                    forward_uniter_outputs = self.forward_uniter(
                        input_ids=batch['input_ids'], position_ids=batch['position_ids'],
                        img_feat=batch['img_feat'], img_pos_feat=batch['img_pos_feat'],
                        attn_masks=batch['attn_masks'], gather_index=batch['gather_index'],
                        compute_loss=False,
                    )
                    hard_batch = self._get_hard_batch(batch, forward_uniter_outputs['rank_scores'], sample_from)
                    self.train()
                forward_uniter_outputs = self.forward_uniter(
                    sample_size=hard_batch['sample_size'],
                    input_ids=hard_batch['input_ids'], position_ids=hard_batch['position_ids'],
                    img_feat=hard_batch['img_feat'], img_pos_feat=hard_batch['img_pos_feat'],
                    attn_masks=hard_batch['attn_masks'], gather_index=hard_batch['gather_index'],
                    compute_loss=True)
                model_outputs['rank_loss'] = forward_uniter_outputs['rank_loss']
                model_outputs['rank_corrects'] = forward_uniter_outputs['rank_corrects']
            else:
                forward_uniter_outputs = self.forward_uniter(
                    input_ids=batch['input_ids'], position_ids=batch['position_ids'],
                    img_feat=batch['img_feat'], img_pos_feat=batch['img_pos_feat'],
                    attn_masks=batch['attn_masks'], gather_index=batch['gather_index'],
                    compute_loss=compute_loss, sigmoid_norm=True)
                model_outputs.update(forward_uniter_outputs)
            return model_outputs
        else:
            if compute_mlm:
                self.train()
                mlm_outputs = self.forward_mlm(
                    input_ids=batch['mlm_input_ids'], position_ids=batch['mlm_position_ids'],
                    img_feat=batch['mlm_img_feat'], img_pos_feat=batch['mlm_img_pos_feat'],
                    attn_masks=batch['mlm_attn_masks'], gather_index=batch['mlm_gather_index'],
                    txt_labels=batch['mlm_txt_labels'], compute_loss=True, sampling=True
                )
                model_outputs['masked_lm_loss'] = mlm_outputs['masked_lm_loss']
                # model_outputs['effect_nsgd_number'] = mlm_outputs['effect_nsgd_number']
                model_outputs['mlm_corrects'] = mlm_outputs['mlm_corrects']
            else:
                with torch.no_grad():
                    self.eval()
                    # print('| mlm_inference | mlm_input_ids: ', batch['mlm_input_ids'].size())
                    mlm_outputs = self.forward_mlm(
                        input_ids=batch['mlm_input_ids'], position_ids=batch['mlm_position_ids'],
                        img_feat=batch['mlm_img_feat'], img_pos_feat=batch['mlm_img_pos_feat'],
                        attn_masks=batch['mlm_attn_masks'], gather_index=batch['mlm_gather_index'],
                        txt_labels=batch['mlm_txt_labels'], compute_loss=False, sampling=True
                    )
            with torch.no_grad():
                # select_indices = mlm_outputs['select_indices']
                nsgd_batch = {}
                nsgd_batch['txt_ids'] = batch['input_ids']
                nsgd_batch['mlm_sample_size'] = len(batch['mlm_position_ids'])
                nsgd_batch['nsgd_sample_size'] = self.nsgd_sample_size
                nsgd_batch['input_ids'] = torch.cat(
                    [batch['input_ids'], mlm_outputs['fill_input_ids']], dim=0)
                nsgd_batch['position_ids'] = torch.cat(
                    [batch['mlm_position_ids'][:1],
                     repeat_interleave(batch['mlm_position_ids'], self.nsgd_sample_size, dim=0)], dim=0)
                nsgd_batch['img_feat'] = torch.cat(
                    [batch['mlm_img_feat'][:1],
                     repeat_interleave(batch['mlm_img_feat'], self.nsgd_sample_size, dim=0)], dim=0)
                nsgd_batch['img_pos_feat'] = torch.cat(
                    [batch['mlm_img_pos_feat'][:1],
                     repeat_interleave(batch['mlm_img_pos_feat'], self.nsgd_sample_size, dim=0)], dim=0)
                nsgd_batch['attn_masks'] = torch.cat(
                    [batch['mlm_attn_masks'][:1],
                     repeat_interleave(batch['mlm_attn_masks'], self.nsgd_sample_size, dim=0)], dim=0)
                nsgd_batch['gather_index'] = torch.cat(
                    [batch['mlm_attn_masks'][:1],
                     repeat_interleave(batch['mlm_gather_index'], self.nsgd_sample_size, dim=0)], dim=0)
                self.eval()
                forward_uniter_outputs = self.forward_uniter(
                    input_ids=nsgd_batch['input_ids'], position_ids=nsgd_batch['position_ids'],
                    img_feat=nsgd_batch['img_feat'], img_pos_feat=nsgd_batch['img_pos_feat'],
                    attn_masks=nsgd_batch['attn_masks'], gather_index=nsgd_batch['gather_index'],
                    compute_loss=False, sigmoid_norm=True
                )
                nsgd_batch = self._get_nsgd_batch(
                    nsgd_batch, scores=forward_uniter_outputs['rank_scores'],
                    clean=compute_loss)
                self.train()
                assert batch['input_ids'].ne(nsgd_batch['input_ids'][0]).long().sum().item() == 0
                model_outputs['effect_nsgd_number'] = nsgd_batch['effect_num']
                model_outputs['rank_adv_scores'] = forward_uniter_outputs['rank_scores_sigmoid']
            if compute_loss:
                forward_uniter_outputs = self.forward_uniter(
                    sample_size=nsgd_batch['sample_size'],
                    input_ids=nsgd_batch['input_ids'], position_ids=nsgd_batch['position_ids'],
                    img_feat=nsgd_batch['img_feat'], img_pos_feat=nsgd_batch['img_pos_feat'],
                    attn_masks=nsgd_batch['attn_masks'], gather_index=nsgd_batch['gather_index'],
                    compute_loss=True
                )
                model_outputs.update(forward_uniter_outputs)
            else:
                model_outputs['nsgd_adv_batch'] = nsgd_batch
            return model_outputs

    def _get_nsgd_batch(self, batch, scores, clean=True):
        batch = defaultdict(lambda: None, batch)
        txt_ids = batch['txt_ids']
        mlm_sample_size, nsgd_sample_size = batch['mlm_sample_size'], batch['nsgd_sample_size']

        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        if clean:
            # print('| clean:', clean)
            c1_penalty_scores = repeat_interleave(txt_ids, mlm_sample_size * nsgd_sample_size + 1, dim=0).\
                ne(input_ids).float().sum(-1).le(0).float().mul(3)
            c2_penalty_scores = input_ids.unsqueeze(1).ne(input_ids.unsqueeze(0)).\
                float().sum(-1).le(0).float().\
                triu(diagonal=1).sum(-1).gt(0).float().mul(2)
            effect_num = c1_penalty_scores.add(c2_penalty_scores).eq(0).long().sum()
            hard_batch['effect_num'] = effect_num
            hard_scores = scores.squeeze(-1).sub(c1_penalty_scores.add(c2_penalty_scores).type_as(scores))
        else:
            hard_batch['effect_num'] = len(scores)
            hard_scores = scores.squeeze(-1)
        # print('| hard_scores: ', hard_scores.size(), self.hard_size)
        hard_indices = hard_scores[1:].topk(self.hard_size, sorted=True)[1] + 1
        # hard_indices = hard_indices[256:]
        hard_size = len(hard_indices)
        indices = torch.cat([torch.zeros(1, dtype=torch.long, device=hard_indices.device),
                             hard_indices])
        # indices = hard_indices

        attention_mask = attention_mask.index_select(0, indices)
        gather_index = gather_index.index_select(0, indices)
        if position_ids.size(0) != 1:
            position_ids = position_ids[:hard_size+1]

        input_ids = input_ids.index_select(0, indices)
        # expect same image features for all pairs
        img_feat = img_feat[:hard_size+1]
        img_pos_feat = img_pos_feat[:hard_size+1]

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['gather_index'] = gather_index

        return hard_batch

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(
            self.hard_size, sorted=False)[1] + 1
        indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                         device=hard_indices.device),
                             hard_indices])

        attention_mask = attention_mask.index_select(0, indices)
        gather_index = gather_index.index_select(0, indices)
        if position_ids.size(0) != 1:
            position_ids = position_ids[:self.hard_size+1]

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(dim=1).max().item()
            max_i = max_len - input_ids.size(1)
            attention_mask = attention_mask[:, :max_len]
            gather_index = gather_index[:, :max_len]
            img_feat = img_feat.index_select(0, indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(0, indices)[:, :max_i, :]
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size+1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(0, indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size+1]
            img_pos_feat = img_pos_feat[:self.hard_size+1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['gather_index'] = gather_index

        return hard_batch
