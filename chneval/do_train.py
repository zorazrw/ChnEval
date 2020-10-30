#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do Pre-training

@author: Zhiruo Wang
"""

import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from loader import LOADERS
from trainer import TRAINERS
from modeling import MODELS
from tokenizer import FullTokenizer
from optimizer import AdamW, WarmupLinearSchedule



def train_and_validate(args):
    tokenizer = FullTokenizer(args.vocab_path)
    args.vocab_size = len(tokenizer.vocab)

    model = MODELS[args.target](args)

    if args.dist_train:   # multiple GPU mode
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu: # single GPU mode
        worker(args.gpu_id, None, args, model)
    else:                 # CPU mode
        worker(None, None, args, model)


def worker(proc_id, gpu_ranks, args, model):
    if args.dist_train:  # multiple GPU mode
        rank = gpu_ranks[proc_id] % args.world_size
        gpu_id = gpu_ranks[proc_id] % args.device_count
    elif args.single_gpu:  # single GPU mode
        rank = None
        gpu_id = proc_id
    else:  # CPU mode
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = LOADERS[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = LOADERS[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)

    if gpu_id is not None: 
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)


    # build optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.total_steps*args.warmup, t_total=args.total_steps)

    if args.dist_train:
        # initialize multiprocessing distributed training environment
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        print("Worker {} is training ... ".format(rank))
    else:
        print("Worker is training ...")
    TRAINERS[args.target](args, gpu_id, rank, train_loader, model, optimizer, scheduler)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # vocabulary options
    parser.add_argument("--vocab_path", type=str, default="../data/vocab.txt", 
        help="Path of the vocabulary file.")
    
    # configuration options
    parser.add_argument("--hidden_size", type=int, default=768, 
        help="Size of the hidden states.")
    parser.add_argument("--intermediate_size", type=int, default=3072, 
        help="Size of the intermediate layer.")
    parser.add_argument("--max_sequence_length", type=int, default=512, 
        help="Maximum length of table string.")
    
    parser.add_argument("--num_attention_heads", type=int, default=12, 
        help="Number of the atttention heads.")
    parser.add_argument("--num_hidden_layers", type=int, default=12, 
        help="Number of the hidden layers.")

    parser.add_argument("--hidden_dropout_prob", type=int, default=0.1, 
        help="Dropout probability for hidden layers.")
    parser.add_argument("--attention_dropout_prob", type=int, default=0.1, 
        help="Dropout probability for attention.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    
    # training options
    parser.add_argument("--batch_size", type=int, default=32, 
        help="Size of the input batch.")
    parser.add_argument("--seq_length", type=int, default=128, 
        help="Length of pre-processed sequences.")
    parser.add_argument("--total_steps", type=int, default=300000, 
        help="Total training steps.")
    parser.add_argument("--report_steps", type=int, default=100, 
        help="Specific steps to print prompt.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=100000, 
        help="Specific steps to save model checkpoint.")
    parser.add_argument("--instances_buffer_size", type=int, default=1000000, 
        help="The buffer size of instances in memory.")

    # io options
    parser.add_argument("--dataset_path", type=str, default="dataset.pt", 
        help="Base path of the preprocessed dataset.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, 
        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str, required=True, 
        help="Path of the output model.")

    # optimizer options
    parser.add_argument("--warmup", type=float, default=0.1, 
        help="Warm up value.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
        help="Initial learning rate.")

    # gpu options
    parser.add_argument("--world_size", type=int, default=1, 
        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, 
        help="List of ranks of each process."
        " Each process has a unique integer rank whose value in the interval [0, world_size], and runs in a single GPU.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", 
        type=str, help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], 
        default="nccl", type=str, help="Distributed backend.")
    
    parser.add_argument("--target", choices=["bert", "mlm", "sbo", "sop"], 
        default="bert", type=str, help="Pre-training target.")
    
    args = parser.parse_args()


    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs." 
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit." 
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs." 
        # multiple GPU mode
        args.dist_train = True
        args.ranks_num = ranks_num
        args.device_count = torch.cuda.device_count()
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs." 
        # single GPU mode.
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id <= torch.cuda.device_count(), "Invalid specified GPU device." 
        args.dist_train = False
        args.single_gpu = True
        print("Using single GPU: {} for training.".format(args.gpu_id))
    else:
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    train_and_validate(args)



if __name__ == "__main__":
    main()
