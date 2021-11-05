"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
from collections import defaultdict
import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
from .sampler import TokenBucketSampler
from collections import Counter


def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def random_word2(tokens, vocab_range, mask, mlm_positions, position_to_prob):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :param mlm_positions: the positions sequence. if select, all tokens in the positions would be masked.
    :param position_to_prob: the sampling probability of each token in the specific position
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    random.shuffle(mlm_positions)
    output_label = [-1] * len(tokens)
    for positions in mlm_positions:
        sample_prob = sum([position_to_prob.get(position, 0.15) for position in positions]) / max(len(positions), 1)
        prob = random.random()
        if prob < sample_prob:
            prob /= sample_prob

            for position in positions:
                token = tokens[position]
                if output_label[position] == -1:
                    prob2 = random.random()
                    if prob2 < 0.8:
                        tokens[position] = mask
                    elif prob2 < 0.9:
                        tokens[position] = random.choice(list(range(*vocab_range)))
                    output_label[position] = token

    if all(o == -1 for o in output_label):
        # at least mask 1
        select_positions = mlm_positions[0]
        for position in select_positions:
            token = tokens[position]
            prob2 = random.random()
            if prob2 < 0.8:
                tokens[position] = mask
            elif prob2 < 0.9:
                tokens[position] = random.choice(list(range(*vocab_range)))
            output_label[position] = token

    return tokens, output_label


class TokenBucketSamplerForItm(TokenBucketSampler):
    def __init__(self, dset, *args, **kwargs):
        super().__init__(dset.lens, *args, **kwargs)
        self.dset = dset

    def __iter__(self):
        it = super().__iter__()
        self.dset.new_epoch()
        self._lens = self.dset.lens
        return it


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class ItmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """
    def __init__(self, txt_db, img_db, neg_sample_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))

        self.neg_sample_p = neg_sample_p
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1-self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            if self.labels[i] == 0:
                img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
            self.train_imgs.append(img_fname)
            self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        img_fname = self.train_imgs[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def itm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len-tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


def itm_ot_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    # OT inputs
    max_tl = max(txt_lens)
    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad}

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'ot_inputs': ot_inputs}
    return batch


class ItmRankDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, \
            "ItmRankDataset need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        # images partitioned by rank
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.img_name_list = list(self.img2txts.keys())

        assert neg_sample_size > 0
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)
        id_pairs.extend([(gt_txt_id, neg_img_id)
                         for neg_img_id in neg_sample_img_ids] +
                        [(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + 2*self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        inputs = []
        for txt_id, img_id in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img_id)
            # mask
            attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks))

        return inputs


def itm_rank_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     ) = map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'sample_size': sample_size}
    return batch


class ItmRankDatasetHardNegFromText(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.img_name_list = list(self.img2txts.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        input_ids = self.txt_db[gt_txt_id]['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        neg_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        img_ids = [gt_img_fname] + neg_img_ids
        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch


class ItmRankDatasetHardNegFromImage(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.txt_name_list = list(self.txt2img.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]
        gt_txt_ids = self.img2txts[gt_img_id]

        # process image features (gt always first)
        img_feat, img_pos_feat, nbb = self._get_img_feat(gt_img_id)
        img_feat = img_feat.unsqueeze(0)
        img_pos_feat = img_pos_feat.unsqueeze(0)

        # sample negative
        neg_txt_ids = sample_negative(
            self.txt_name_list, gt_txt_ids, self.neg_sample_size)
        txt_ids = [gt_txt_id] + neg_txt_ids

        # process text inputs
        all_inputs = []
        txt_lens = []
        for txt_id in txt_ids:
            input_ids = self.txt_db.combine_inputs(
                self.txt_db[txt_id]['input_ids'])
            all_inputs.append(input_ids)
            txt_lens.append(len(input_ids))
        input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        attn_masks = torch.zeros(len(txt_ids), max(txt_lens) + nbb).long()
        for i, tl in enumerate(txt_lens):
            attn_masks.data[i, :tl+nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index(txt_lens, [nbb]*len(txt_ids),
                                        len(txt_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch


def itm_rank_hn_collate(inputs):
    assert len(inputs) == 1
    return inputs[0]


class ItmValDataset(DetectFeatTxtTokDataset):
    """ For evaluating Image-Text-Retrieval task """
    def __init__(self, db_dir, img_dir, mini_batch_size=400, mlm_sample_size=1):
        super().__init__(db_dir, img_dir)

        self.txt_lens = self.lens[:]
        del self.lens
        self.txt2img = self.txt_db.txt2img
        self.img2txts = self.txt_db.img2txts
        self.all_img_ids = list(self.img2txts.keys())
        self.id2len = self.txt_db.id2len
        self.i2len = {i: self.id2len.get(self.ids[i]) for i in range(len(self.ids))}

        assert len(self.img2txts) >= mini_batch_size > 0
        self.bs = mini_batch_size
        self.mlm_sample_size = mlm_sample_size

    def _get_batch_ids(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]

        # sample fixed negatives for each gt image
        i = self.all_img_ids.index(gt_img_id)
        neg_st = i+1
        neg_end = neg_st+self.bs-1
        if neg_end > len(self.all_img_ids):
            # warp around
            neg_end -= len(self.all_img_ids)
            neg_img_ids = (self.all_img_ids[neg_st:]
                           + self.all_img_ids[:neg_end])
        else:
            neg_img_ids = self.all_img_ids[neg_st:neg_end]

        assert len(neg_img_ids) == (self.bs - 1),\
            "Did not sample enough neg samples"

        return gt_img_id, neg_img_ids

    def __getitem__(self, i):
        """ this returns list of mini-batches """
        gt_img_id, neg_img_ids = self._get_batch_ids(i)
        # NOTE 1st one is gt img
        batch = self.get_batch(i, [gt_img_id] + neg_img_ids)
        return batch

    def create_mlm_io(self, input_ids, nsample=1):
        mlm_input_ids, mlm_txt_labels = [], []
        for i in range(nsample):
            # tokens, vocab_range, mask, mlm_positions, position_to_prob
            r_input_ids, txt_labels = random_word(
                copy.copy(input_ids), self.txt_db.v_range, self.txt_db.mask)
            mlm_input_ids.append(torch.tensor([self.txt_db.cls_] + r_input_ids + [self.txt_db.sep]))
            mlm_txt_labels.append(torch.tensor([-1] + txt_labels + [-1]))
        mlm_input_ids = torch.stack(mlm_input_ids, dim=0)
        mlm_txt_labels = torch.stack(mlm_txt_labels, dim=0)
        return mlm_input_ids, mlm_txt_labels

    def create_mlm_io2(self, input_ids, tree=None, nsample=1):
        mlm_input_ids, mlm_txt_labels = [], []
        sample_prob = 0.15

        mlm_positions = [[i] for i in range(len(input_ids))]
        for struct_type in ['relation', 'attribute', 'node']:
            struct_nodes = tree.get(struct_type)
            for struct_node in struct_nodes:
                positions = struct_node.get('ids')
                if positions is not None:
                    mlm_positions.append(positions)
        # mlm_positions = list(set(mlm_positions))
        position_counter = Counter()
        for positions in mlm_positions:
            position_counter.update(positions)
        position_to_prob = {position: sample_prob / max(freq, 1.0) for position, freq in position_counter.items()}

        # print("| mlm_positions: ", mlm_positions)
        for i in range(nsample):
            r_input_ids, txt_labels = random_word2(
                copy.copy(input_ids), self.txt_db.v_range, self.txt_db.mask,
                mlm_positions=mlm_positions, position_to_prob=position_to_prob)
            mlm_input_ids.append(torch.tensor([self.txt_db.cls_] + r_input_ids + [self.txt_db.sep]))
            mlm_txt_labels.append(torch.tensor([-1] + txt_labels + [-1]))
        mlm_input_ids = torch.stack(mlm_input_ids, dim=0)
        mlm_txt_labels = torch.stack(mlm_txt_labels, dim=0)
        return mlm_input_ids, mlm_txt_labels

    def get_batch(self, i, img_ids, forward_mlm=False):
        batch = {}

        example = super().__getitem__(i)

        input_ids = example['input_ids']

        if forward_mlm:
            gt_txt_id = self.ids[i]
            gt_img_fname = self.txt2img[gt_txt_id]
            img_ids = [gt_img_fname]
            # Process Masked Text to Generate Adversarial Samples
            # mlm_input_ids, mlm_txt_labels = self.create_mlm_io(input_ids, nsample=self.mlm_sample_size)

            tree = self.txt_db[gt_txt_id]['tree']
            mlm_input_ids, mlm_txt_labels = self.create_mlm_io2(input_ids, tree=tree, nsample=self.mlm_sample_size)

            mlm_position_ids = torch.arange(0, mlm_input_ids.size(1), dtype=torch.long).\
                unsqueeze(0).expand(self.mlm_sample_size, -1)
            img_feat, img_pos_feat, num_bbs = self._get_img_feat(gt_img_fname)
            mlm_img_feat = img_feat.unsqueeze(dim=0).expand(self.mlm_sample_size, *list(img_feat.size()))
            mlm_img_pos_feat = img_pos_feat.unsqueeze(dim=0).expand(self.mlm_sample_size, *list(img_pos_feat.size()))
            tl = mlm_input_ids.size(1)
            mlm_attn_masks = torch.zeros(self.mlm_sample_size, tl+num_bbs).long()
            mlm_attn_masks.data[:, :tl+num_bbs].fill_(1)
            mlm_gather_index = get_gather_index(
                [tl]*self.mlm_sample_size, [num_bbs]*self.mlm_sample_size, self.mlm_sample_size, tl, tl+num_bbs)

            batch['mlm_input_ids'] = mlm_input_ids
            batch['mlm_position_ids'] = mlm_position_ids
            batch['mlm_img_feat'] = mlm_img_feat
            batch['mlm_img_pos_feat'] = mlm_img_pos_feat
            batch['mlm_attn_masks'] = mlm_attn_masks
            batch['mlm_gather_index'] = mlm_gather_index
            batch['mlm_txt_labels'] = mlm_txt_labels

        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch['input_ids'] = input_ids
        batch['position_ids'] = position_ids
        batch['img_feat'] = img_feat
        batch['img_pos_feat'] = img_pos_feat
        batch['attn_masks'] = attn_masks
        batch['gather_index'] = gather_index
        return batch


def itm_val_collate(inputs):
    assert len(inputs) == 1, "input batch size > 1"
    return inputs[0]


class ItmEvalDataset(ItmValDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st+self.bs]))
        return mini_batches


class ItmAdvEvalDataset(ItmValDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])
        self.all_img_ids = np.array(self.all_img_ids)

    def __getitem__(self, i):
        return self.get_batch(i, [], forward_mlm=True)


class ItmDCEvalDataset(ItmValDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])

    def __getitem__(self, i):
        return self.get_batch(i, [], forward_mlm=True)

    def get_by_img_index(self, i, image_indices):
        mini_batch = self.get_batch(i, self.all_img_ids[image_indices])
        mini_batch_img_txt_input_ids = []
        for image_index in image_indices:
            img_id = self.all_img_ids[image_index]
            txt_ids = self.img2txts[img_id]
            img_txt_input_ids = []
            for txt_id in txt_ids:
                input_ids = self.txt_db[txt_id]['input_ids']
                img_txt_input_ids.extend(input_ids)
            mini_batch_img_txt_input_ids.append(torch.tensor(list(set(mini_batch_img_txt_input_ids))))
        mini_batch['img_txt_input_ids'] = mini_batch_img_txt_input_ids
        mini_batches = [mini_batch]
        return mini_batches


class ItmStaticDataAttackEvalDataset(DetectFeatTxtTokDataset):
    def __init__(self, db_dir, img_dir, mini_batch_size=400, mlm_sample_size=1):
        super().__init__(db_dir, img_dir)

        self.txt_lens = self.lens[:]
        del self.lens
        self.txt2img = self.txt_db.txt2img
        self.img2txts = self.txt_db.img2txts
        self.all_img_ids = list(self.img2txts.keys())
        self.id2len = self.txt_db.id2len
        self.i2len = {i: self.id2len.get(self.ids[i]) for i in range(len(self.ids))}

        assert len(self.img2txts) >= mini_batch_size > 0
        self.bs = mini_batch_size
        self.mlm_sample_size = mlm_sample_size

    def __getitem__(self, i):
        return self.get_batch(i, [], forward_mlm=True)

    def get_batch(self, i, img_ids, forward_mlm=False):
        batch = {}
        example = super().__getitem__(i)
        input_ids = example['input_ids']
        all_attack_text_ids = example['attack_data'][0]
        all_attack_text_ids = all_attack_text_ids[:400]

        all_inputs = [self.txt_db.combine_inputs(token_ids) for token_ids in [input_ids] + all_attack_text_ids]
        txt_lens = [len(token_ids) for token_ids in all_inputs]
        input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
        # print('| inputs_ids: ', len(all_attack_text_ids), input_ids.size())
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]

        # process image features (gt always first)
        img_feat, img_pos_feat, nbb = self._get_img_feat(gt_img_id)
        img_feat = img_feat.unsqueeze(0)
        img_pos_feat = img_pos_feat.unsqueeze(0)
        # print("| img_feat: ", img_feat.size(), img_pos_feat.size())

        attn_masks = torch.zeros(len(txt_lens), max(txt_lens) + nbb).long()
        for i, tl in enumerate(txt_lens):
            attn_masks.data[i, :tl+nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index(txt_lens, [nbb]*len(txt_lens), len(txt_lens), tl, out_size)

        batch['input_ids'] = input_ids
        batch['position_ids'] = position_ids
        batch['img_feat'] = img_feat
        batch['img_pos_feat'] = img_pos_feat
        batch['attn_masks'] = attn_masks
        batch['gather_index'] = gather_index
        return batch


itm_eval_collate = itm_val_collate
