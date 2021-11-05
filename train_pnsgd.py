"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for Image-Text Retrieval with hard negatives
"""
import argparse
import os
from os.path import exists, join
from time import time
import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm

from data import (PrefetchLoader, TxtTokLmdb, ImageLmdbGroup,
                  PNSGDFromText, PNSGDFromImage, pnsgd_collate, itm_rank_hn_collate,
                  ItmValDataset, itm_val_collate,
                  ItmEvalDataset, itm_eval_collate)
from model.nsgd import UniterForNSGD
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM
from utils.itm_eval import evaluate


def build_dataloader(dataset, collate_fn, is_train, opts):
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_lr_scheduler(opts, num_training_steps, optimizer):
    num_warmup_steps = (
        opts.warmup_steps
        if opts.warmup_steps > 0
        else opts.ceil(num_training_steps * opts.args.warmup_ratio)
    )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


def load_optimizer_and_scheduler(optimizer, lr_scheduler, checkpoint):
    """If optimizer and scheduler states exist, load them."""
    if checkpoint is None:
        return

    if os.path.isfile(os.path.join(checkpoint, "optimizer.pt")) and os.path.isfile(
        os.path.join(checkpoint, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location=device)
        )
        lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, "scheduler.pt")))
        return optimizer, lr_scheduler


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    set_random_seed(opts.seed)

    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        # store ITM predictions
        os.makedirs(join(opts.output_dir, 'results_val'))
        os.makedirs(join(opts.output_dir, 'results_test'))
        os.makedirs(join(opts.output_dir, 'results_train'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # train_examples = None
    LOGGER.info(f"Loading Train Dataset {opts.train_txt_dbs}, "
                f"{opts.train_img_dbs}")
    # check multiple DBs
    assert len(opts.train_txt_dbs) == len(opts.train_img_dbs), \
        "train txt_db and img_db have different length"

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
    train_datasets_t = []
    train_datasets_i = []
    for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
        img_db = all_img_dbs[img_path]
        txt_db = TxtTokLmdb(txt_path, opts.max_txt_len)
        train_datasets_t.append(
            PNSGDFromText(txt_db, img_db, opts.negative_size, opts.mlm_sample_size))
        train_datasets_i.append(
            PNSGDFromImage(txt_db, img_db, opts.negative_size))
    train_dataset_t = ConcatDataset(train_datasets_t)
    train_dataset_i = ConcatDataset(train_datasets_i)
    train_dataloader_t = build_dataloader(
        train_dataset_t, pnsgd_collate, True, opts)
    train_dataloader_i = build_dataloader(
        train_dataset_i, pnsgd_collate, True, opts)

    # val
    LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_img_db}")
    val_img_db = all_img_dbs[opts.val_img_db]
    val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
    val_dataset = ItmValDataset(val_txt_db, val_img_db,
                                opts.inf_minibatch_size)
    val_dataloader = build_dataloader(val_dataset, itm_val_collate,
                                      False, opts)
    # eval
    LOGGER.info(f"Loading val, test Dataset for full evaluation: "
                f"{opts.val_txt_db}, {opts.val_img_db}"
                f"{opts.test_txt_db}, {opts.test_img_db}")
    eval_dataset_val = ItmEvalDataset(val_txt_db, val_img_db,
                                      opts.inf_minibatch_size)
    eval_loader_val = build_dataloader(eval_dataset_val, itm_eval_collate,
                                       False, opts)
    test_img_db = all_img_dbs[opts.test_img_db]
    test_txt_db = TxtTokLmdb(opts.test_txt_db, -1)
    eval_dataset_test = ItmEvalDataset(test_txt_db, test_img_db,
                                       opts.inf_minibatch_size)
    eval_loader_test = build_dataloader(eval_dataset_test, itm_eval_collate,
                                        False, opts)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    model = UniterForNSGD.from_pretrained(
        opts.model_config, state_dict=checkpoint,
        img_dim=IMG_DIM, margin=opts.margin, hard_size=opts.hard_neg_size,
        nsgd_sample_size=opts.nsgd_sample_size, nsgd_sample_temperature=opts.nsgd_sample_temperature)
    model.init_output()  # pretrain ITM head is different from ranking head
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    # print('| optimizer file:', os.path.join(checkpoint, "optimizer.pt"))
    # if os.path.exists(os.path.join(checkpoint, "optimizer.pt")) and \
    #         os.path.isfile(os.path.join(checkpoint, "optimizer.pt")):
    #     # Load in optimizer and scheduler states
    #     device = 'gpu' if torch.cuda.is_available() else 'cpu'
    #     optimizer.load_state_dict(torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location=device))
    model, optimizer = amp.initialize(model, optimizer, enabled=opts.fp16, opt_level='O2')
    # Prepare scheduler
    # lr_scheduler = build_lr_scheduler(opts, opts.num_train_steps, optimizer)
    # print('| scheduler file:', os.path.join(checkpoint, "scheduler.pt"))
    # if os.path.exists(os.path.join(checkpoint, "scheduler.pt")) and \
    #         os.path.isfile(os.path.join(checkpoint, "scheduler.pt")):
    #     lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, 'scheduler.pt')))

    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d",
                sum(all_gather_list(len(train_dataset_t))))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # running_loss = RunningMeter('loss')
    running_loss_dict = {loss: RunningMeter(loss) for loss in
                         ['mlm_loss', 'nsgd_rank_loss', 'i2t_hn_rank_loss', 't2i_hn_rank_loss']}
    model.train()

    global_step = 0
    step = 0
    n_examples = 0
    n_hard_ex = 0
    start = time()
    train_iter_i = iter(train_dataloader_i)
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    effect_nsgd_number, mlm_corrects, nsgd_rank_corrects, i2t_hn_rank_corrects, t2i_hn_rank_corrects = \
        [], [[], []], [[], []], [[], []], [[], []]
    incremental_adv_scores = []
    while True:
        for batch in train_dataloader_t:

            # hard text from image
            try:
                batch_i = next(train_iter_i)
            except StopIteration:
                train_iter_i = iter(train_dataloader_i)
                batch_i = next(train_iter_i)

            n_examples += batch['mlm_attn_masks'].size(0)
            model_outputs = model(
                batch, sample_from='gi', compute_loss=True,
                compute_mlm=True if isinstance(args.mlm_lambda, float) and args.mlm_lambda > 0 else False)
            loss = 0.0
            if model_outputs.get('masked_lm_loss') is not None:
                mlm_loss = model_outputs.get('masked_lm_loss').mean()
                # print('| mlm_loss: ', mlm_loss)
                loss += args.mlm_lambda * mlm_loss
                running_loss_dict.get('mlm_loss')(mlm_loss.item())
                mlm_corrects[0].append(model_outputs['mlm_corrects'].sum().item())
                mlm_corrects[1].append(model_outputs['mlm_corrects'].numel())
                # print('mlm_corrects:', mlm_corrects[0][-1]/mlm_corrects[1][-1])
                effect_nsgd_number.append(model_outputs['effect_nsgd_number'])
            if model_outputs.get('rank_loss') is not None:
                rank_adv_scores = model_outputs['rank_adv_scores'].squeeze(-1)
                # print('| rank_adv_scores: ', rank_adv_scores.min(), rank_adv_scores.max())
                incremental_adv_score = rank_adv_scores[1:] - rank_adv_scores[0]
                incremental_adv_scores.append(incremental_adv_score)
                nsgd_rank_loss = model_outputs.get('rank_loss')
                rank_corrects = model_outputs.get('rank_corrects')
                nsgd_hard_ex = rank_corrects.numel()
                nsgd_rank_corrects[0].append(rank_corrects.sum().item())
                nsgd_rank_corrects[1].append(nsgd_hard_ex)
                # print('nsgd_rank_corrects:', nsgd_rank_corrects[0][-1]/nsgd_rank_corrects[1][-1])
                n_hard_ex += nsgd_hard_ex
                nsgd_rank_loss = nsgd_rank_loss.mean()
                running_loss_dict.get('nsgd_rank_loss')(nsgd_rank_loss.item())
                loss += args.nsgd_rank_lambda * nsgd_rank_loss.mean()
            if isinstance(loss, torch.Tensor):
                loss = loss / opts.train_batch_size
                with amp.scale_loss(loss, optimizer, delay_unscale=True, # loss_id=0
                                    ) as scaled_loss:
                    scaled_loss.backward()

            n_examples += batch_i['attn_masks'].size(0)
            model_outputs = model(batch_i, sample_from='i', compute_loss=True)
            loss = model_outputs.get('rank_loss')
            rank_corrects = model_outputs.get('rank_corrects')
            i2t_hard_ex = rank_corrects.numel()
            i2t_hn_rank_corrects[0].append(rank_corrects.sum().item())
            i2t_hn_rank_corrects[1].append(i2t_hard_ex)
            n_hard_ex += i2t_hard_ex
            loss = loss.mean() / opts.train_batch_size
            with amp.scale_loss(loss, optimizer, delay_unscale=True, # loss_id=1
                                ) as scaled_loss:
                scaled_loss.backward()
            running_loss_dict.get('t2i_hn_rank_loss')(loss.item())

            # hard image from text
            n_examples += batch['attn_masks'].size(0)
            model_outputs = model(batch, sample_from='t', compute_loss=True)
            loss = model_outputs.get('rank_loss')
            rank_corrects = model_outputs.get('rank_corrects')
            t2i_hard_ex = rank_corrects.numel()
            t2i_hn_rank_corrects[0].append(rank_corrects.sum().item())
            t2i_hn_rank_corrects[1].append(t2i_hard_ex)
            n_hard_ex += t2i_hard_ex
            # NOTE we use gradient accumulation to implemented train_batch_size
            loss = loss.mean() / opts.train_batch_size

            step += 1
            delay_unscale = step % opts.train_batch_size != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale, #loss_id=2
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))
            running_loss_dict.get('i2t_hn_rank_loss')(loss.item())

            if step % opts.train_batch_size == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                incremental_adv_scores = torch.cat(incremental_adv_scores, dim=0)
                TB_LOGGER.add_histogram(
                    'incremental_adv_score', incremental_adv_scores.data.cpu().numpy(), global_step)
                incremental_adv_scores = []

                TB_LOGGER.add_scalar(
                    'mlm_loss', running_loss_dict['mlm_loss'].val, global_step)
                TB_LOGGER.add_scalar(
                    'nsgd_rank_loss', running_loss_dict['nsgd_rank_loss'].val, global_step)
                TB_LOGGER.add_scalar(
                    'i2t_hn_rank_loss', running_loss_dict['i2t_hn_rank_loss'].val, global_step)
                TB_LOGGER.add_scalar(
                    't2i_hn_rank_loss', running_loss_dict['t2i_hn_rank_loss'].val, global_step)
                TB_LOGGER.add_scalar(
                    'nsgd_rank_cr', sum(nsgd_rank_corrects[0]) / float(sum(nsgd_rank_corrects[1])), global_step)
                TB_LOGGER.add_scalar(
                    'i2t_hn_rank_cr', sum(i2t_hn_rank_corrects[0]) / float(sum(i2t_hn_rank_corrects[1])), global_step)
                TB_LOGGER.add_scalar(
                    't2i_hn_rank_cr', sum(t2i_hn_rank_corrects[0]) / float(sum(t2i_hn_rank_corrects[1])), global_step)
                TB_LOGGER.add_scalar(
                    'effect_nsgd_number', sum(effect_nsgd_number) / float(len(effect_nsgd_number)), global_step)
                TB_LOGGER.step()
                effect_nsgd_number, mlm_corrects, nsgd_rank_corrects, i2t_hn_rank_corrects, t2i_hn_rank_corrects = \
                    [], [[], []], [[], []], [[], []], [[], []]

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 50 == 0:
                    # monitor training throughput
                    LOGGER.info(f'------------Step {global_step}-------------')
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time()-start))
                    tot_hn = sum(all_gather_list(n_hard_ex))
                    hn_per_sec = int(tot_hn / (time()-start))
                    LOGGER.info(f'{tot_ex} ({tot_hn}) examples (hard) '
                                f'trained at {ex_per_sec} ({hn_per_sec}) ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
                    TB_LOGGER.add_scalar('perf/hn_per_s',
                                         hn_per_sec, global_step)
                    LOGGER.info(f'-------------------------------------------')

                if global_step % opts.valid_steps == 0:
                    if opts.full_val:
                        LOGGER.info(
                            f"========================== Step {global_step} "
                            f"==========================")
                        val_log = evaluate(model, eval_loader_val)
                        try:
                            TB_LOGGER.log_scaler_dict(
                                {f"valid/{k}": v for k, v in val_log.items()})
                            LOGGER.info(f"image retrieval R1: "
                                        f"{val_log['img_r1']*100:.2f},\n"
                                        f"image retrieval R5: "
                                        f"{val_log['img_r5']*100:.2f},\n"
                                        f"image retrieval R10: "
                                        f"{val_log['img_r10']*100:.2f}\n"
                                        f"text retrieval R1: "
                                        f"{val_log['txt_r1']*100:.2f},\n"
                                        f"text retrieval R5: "
                                        f"{val_log['txt_r5']*100:.2f},\n"
                                        f"text retrieval R10: "
                                        f"{val_log['txt_r10']*100:.2f}")
                            LOGGER.info("================================="
                                        "=================================")
                        except KeyError:
                            pass
                    else:
                        val_log = validate(model, val_dataloader)
                        TB_LOGGER.log_scaler_dict(val_log)
                    model_saver.save(model, global_step, optimizer=optimizer)
                    # torch.save(optimizer.state_dict(), os.path.join())
                    # torch.save(lr_scheduler.state_dict(), os.path.join())

            if global_step >= opts.num_train_steps:
                break

        if global_step >= opts.num_train_steps:
            break

    pbar.close()
    # final validation
    val_log = validate(model, val_dataloader)
    TB_LOGGER.log_scaler_dict(val_log)
    model_saver.save(model, f'{global_step}_final')

    # evaluation
    for split, loader in [('val', eval_loader_val),
                          ('test', eval_loader_test)]:
        eval_log = evaluate(model, loader)
        TB_LOGGER.log_scaler_dict({f"eval/{split}_{k}": v
                                   for k, v in eval_log.items()})
        if hvd.rank() != 0:
            continue
        LOGGER.info(
            f"========================= {split} ===========================\n"
            f"image retrieval R1: {eval_log['img_r1']*100:.2f},\n"
            f"image retrieval R5: {eval_log['img_r5']*100:.2f},\n"
            f"image retrieval R10: {eval_log['img_r10']*100:.2f}\n"
            f"text retrieval R1: {eval_log['txt_r1']*100:.2f},\n"
            f"text retrieval R5: {eval_log['txt_r5']*100:.2f},\n"
            f"text retrieval R10: {eval_log['txt_r10']*100:.2f}")
    LOGGER.info("=========================================================")


@torch.no_grad()
def validate(model, val_loader):
    if hvd.rank() == 0:
        pbar = tqdm(total=len(val_loader))
    else:
        pbar = NoOp()
    LOGGER.info("start running Image Retrieval validation ...")
    model.eval()
    n_ex = 0
    st = time()

    recall_at_1, recall_at_5, recall_at_10 = 0, 0, 0
    for batch in val_loader:
        model_outputs = model(batch, compute_loss=False)
        if isinstance(model_outputs, dict):
            scores = model_outputs['rank_scores']
        else:
            scores = model_outputs
        _, indices = scores.squeeze(1).topk(10, dim=0)
        rank = (indices == 0).nonzero()
        if rank.numel():
            rank = rank.item()
            if rank < 1:
                recall_at_1 += 1
            if rank < 5:
                recall_at_5 += 1
            if rank < 10:
                recall_at_10 += 1
        n_ex += 1
        pbar.update(1)
    n_ex = sum(all_gather_list(n_ex))
    recall_at_1 = sum(all_gather_list(recall_at_1)) / n_ex
    recall_at_5 = sum(all_gather_list(recall_at_5)) / n_ex
    recall_at_10 = sum(all_gather_list(recall_at_10)) / n_ex
    tot_time = time()-st
    val_log = {'valid/ex_per_s': n_ex/tot_time,
               'valid/recall_1': recall_at_1,
               'valid/recall_5': recall_at_5,
               'valid/recall_10': recall_at_10}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"recall_1: {recall_at_1*100:.2f}, "
                f"recall_5: {recall_at_5*100:.2f}, "
                f"recall_10: {recall_at_10*100:.2f}")
    pbar.close()
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained MLM")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="batch size (# positive examples) for training. "
                             "(implemented with gradient accumulation)")

    parser.add_argument("--negative_size", default=511, type=int,
                        help="Number of negative samples per positive sample"
                             "(forward only)")
    parser.add_argument("--hard_neg_size", default=31, type=int,
                        help="Number of hard negative samples "
                             "per positive sample (acutally used to train)")
    parser.add_argument("--mlm_sample_size", default=22, type=int,
                        help="Number of samples following masked language masking"
                             "per positive sample (acutally used to train)")
    parser.add_argument("--nsgd_sample_size", default=22, type=int,
                        help="Number of NSGD for each mlm sample"
                             "per positive sample (acutally used to train)")
    parser.add_argument("--nsgd_sample_temperature", default=2.0, type=float,
                        help="sampling temperature of NSGD sampling. ")

    parser.add_argument("--mlm_lambda", default=0.1, type=float,
                        help="lambda in training of mask language modeling")
    parser.add_argument("--nsgd_rank_lambda", default=1.0, type=float,
                        help="lambda in training of NSGD ranking loss")
    parser.add_argument("--margin", default=0.2, type=float,
                        help="margin of ranking loss")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=0.25, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--full_val', action='store_true',
                        help="Always run full evaluation during training")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    # for tensor core
    assert (args.negative_size+1) % 8 == (args.hard_neg_size+1) % 8 == 0

    print('| args: ', args)
    main(args)
