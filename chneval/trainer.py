#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Trainers

@author: Zhiruo Wang
"""

import time
import torch


def save_model(model, save_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
        
        
def train_bert(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_mlm_loss, total_nsp_loss = 0., 0., 0.
    total_mlm_correct, total_mlm_count = 0., 0.
    total_nsp_correct, total_nsp_count = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        input_ids, segment_pos, attn_mask, mlm_labels, nsp_labels = next(loader_iter)
        while input_ids.size()[0] == 0 or torch.min(mlm_labels) >= 0:
            input_ids, segment_pos, attn_mask, mlm_labels, nsp_labels = next(loader_iter)

        model.zero_grad()
        if gpu_id is not None:
            input_ids = input_ids.cuda(gpu_id)
            mlm_labels = mlm_labels.cuda(gpu_id)
            nsp_labels = nsp_labels.cuda(gpu_id)
            segment_pos = segment_pos.cuda(gpu_id)
            attn_mask = attn_mask.cuda(gpu_id)
              
        # forward
        mlm_triple, nsp_triple = model(input_ids, segment_pos, attn_mask, mlm_labels, nsp_labels)
        mlm_loss, mlm_correct, mlm_count = mlm_triple
        nsp_loss, nsp_correct, nsp_count = nsp_triple
        
        # backward
        loss = mlm_loss + nsp_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
        total_mlm_correct += mlm_correct.item()
        total_nsp_correct += nsp_correct.item()
        total_mlm_count += mlm_count.item()
        total_nsp_count += nsp_count
        
        if steps % args.report_steps == 0  and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * input_ids.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * input_ids.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_nsp: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_nsp: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    total_loss / args.report_steps, 
                    total_mlm_loss / args.report_steps,
                    total_nsp_loss / args.report_steps,
                    total_mlm_correct / total_mlm_count,
                    total_nsp_correct / total_nsp_count))
            
            total_loss, total_mlm_loss, total_nsp_loss = 0., 0., 0.
            total_mlm_correct, total_mlm_count = 0., 0.
            total_nsp_correct, total_nsp_count = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1



def train_sop(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_mlm_loss, total_sop_loss = 0., 0., 0.
    total_mlm_correct, total_mlm_count = 0., 0.
    total_sop_correct, total_sop_count = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        input_ids, segment_pos, attn_mask, mlm_labels, sop_labels = next(loader_iter)

        model.zero_grad()
        if gpu_id is not None:
            input_ids = input_ids.cuda(gpu_id)
            mlm_labels = mlm_labels.cuda(gpu_id)
            sop_labels = sop_labels.cuda(gpu_id)
            segment_pos = segment_pos.cuda(gpu_id)
            attn_mask = attn_mask.cuda(gpu_id)
              
        # forward
        mlm_triple, sop_triple = model(input_ids, segment_pos, attn_mask, mlm_labels, sop_labels)
        mlm_loss, mlm_correct, mlm_count = mlm_triple
        sop_loss, sop_correct, sop_count = sop_triple
        
        # backward
        loss = mlm_loss + sop_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_sop_loss += sop_loss.item()
        total_mlm_correct += mlm_correct.item()
        total_sop_correct += sop_correct.item()
        total_mlm_count += mlm_count.item()
        total_sop_count += sop_count
        
        if steps % args.report_steps == 0  and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * input_ids.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * input_ids.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_nsp: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_nsp: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    total_loss / args.report_steps, 
                    total_mlm_loss / args.report_steps,
                    total_sop_loss / args.report_steps,
                    total_mlm_correct / total_mlm_count,
                    total_sop_correct / total_sop_count))
            
            total_loss, total_mlm_loss, total_sop_loss = 0., 0., 0.
            total_mlm_correct, total_mlm_count = 0., 0.
            total_sop_correct, total_sop_count = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1



def train_mlm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_correct, total_count = 0., 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        input_ids, segment_pos, attn_mask, mlm_labels = next(loader_iter)

        model.zero_grad()
        if gpu_id is not None:
            input_ids = input_ids.cuda(gpu_id)
            segment_pos = segment_pos.cuda(gpu_id)
            attn_mask = attn_mask.cuda(gpu_id)
            mlm_labels = mlm_labels.cuda(gpu_id)
        
        # Forward.
        loss, correct, count = model(input_ids, segment_pos, attn_mask, mlm_labels)
        
        # Backward.
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_correct += correct.item()
        total_count += count.item()
        
        if steps % args.report_steps == 0  and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * input_ids.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * input_ids.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss_mlm: {:3.3f}"
                  "| acc_mlm: {:3.3f}".format(
                    steps, total_steps, 
                    done_tokens / elapsed, 
                    total_loss / args.report_steps,
                    total_correct / total_count))
            
            total_loss, total_correct, total_count = 0., 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1



def train_sbo(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_mlm_loss, total_sbo_loss = 0., 0., 0.
    total_mlm_correct, total_mlm_count = 0., 0.
    total_sbo_correct, total_sbo_count = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        input_ids, token_type_ids, attn_mask, mlm_labels = next(loader_iter)

        model.zero_grad()
        if gpu_id is not None:
            input_ids = input_ids.cuda(gpu_id)
            token_type_ids = token_type_ids.cuda(gpu_id)
            attn_mask = attn_mask.cuda(gpu_id)
            mlm_labels = mlm_labels.cuda(gpu_id)
        
        # Forward.
        mlm_triple, sbo_triple = model(input_ids, token_type_ids, attn_mask, mlm_labels)
        mlm_loss, mlm_correct, mlm_count = mlm_triple
        sbo_loss, sbo_correct, sbo_count = sbo_triple
        
        # Backward.
        loss = mlm_loss + sbo_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_sbo_loss += sbo_loss.item()
        total_mlm_correct += mlm_correct.item()
        total_sbo_correct += sbo_correct.item()
        total_mlm_count += mlm_count.item()
        total_sbo_count += sbo_count.item()
        
        if steps % args.report_steps == 0  and (not args.dist_train or (args.dist_train and rank == 0)):
            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * input_ids.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * input_ids.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_sbo: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_sbo: {:3.3f}".format(
                    steps, total_steps, 
                    done_tokens / elapsed, 
                    total_loss / args.report_steps,
                    total_mlm_loss / args.report_steps,
                    total_sbo_loss / args.report_steps,
                    total_mlm_correct / total_mlm_count,
                    total_sbo_correct / total_sbo_count))
            
            total_loss, total_mlm_loss, total_sbo_loss = 0., 0., 0.
            total_mlm_correct, total_sbo_correct = 0., 0.
            total_mlm_count, total_sbo_count = 0., 0.
            
            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1



TRAINERS = {
    "bert": train_bert, 
    "mlm": train_mlm, 
    "sbo": train_sbo, 
    "sop": train_sop
}
