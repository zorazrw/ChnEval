#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loaders

@author: ZhiruoWang
"""

import torch
import random
import pickle



class BertDataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=True):
        self.proc_id = proc_id
        self.shuffle = shuffle
        self.proc_num = proc_num
        self.batch_size = batch_size
        self.buffer_size = args.instances_buffer_size
        # We only need to read dataset once when buffer is big enough to load entire dataset.
        self.repeat_read_dataset = False
        self.f_read = open(dataset_path, "rb")
        self.start = 0
        self.end = 0
        self.data_index = 0
        self.buffer = []
        
    def _fill_buf(self):
        if len(self.buffer) > 0 and not self.repeat_read_dataset:
            if self.shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
        else:
            self.buffer = []
            while True:
                try:
                    instances = pickle.load(self.f_read)
                except EOFError:
                    # Reach file end.
                    if not self.repeat_read_dataset:
                        # Buffer is big enough to load entire dataset.
                        break
                    # Buffer is not big enough, read dataset form start.
                    self.f_read.seek(0)
                    self.data_index = 0
                    instances = pickle.load(self.f_read)

                if self.data_index % self.proc_num == self.proc_id:
                    self.buffer.extend(instances)
                self.data_index += 1

                if len(self.buffer) > self.buffer_size:
                    self.repeat_read_dataset = True 
                    break

            if self.shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)

    def _empty(self):
        return self.start + self.batch_size >= self.end

    def __del__(self):
        self.f_read.close()

    def __iter__(self):
        while True:
            if self._empty():
                self._fill_buf()
            if not self.buffer:
                print("Warning: worker %d data buffer is empty." % self.proc_id)

            instances = self.buffer[self.start : self.start+self.batch_size] 
            self.start += self.batch_size
                
            input_ids, mlm_labels, nsp_label, segment_pos = [], [], [], []
            attn_mask = []
            for ins in instances:
                input_ids.append(ins[0])
                mlm_labels.append([0 for _ in ins[0]])
                for index,replace_id in ins[1]:
                    mlm_labels[-1][index] = replace_id
                nsp_label.append(ins[2])
                segment_pos.append([0 for _ in ins[0]])
                for i in range(ins[3][0], ins[3][1]):
                    segment_pos[-1][i] = 1
                attn_mask.append([0 for _ in ins[0]])
                for i in range(ins[3][1]):
                    attn_mask[-1][i] = 1
            
            yield (
                torch.LongTensor(input_ids), 
                torch.LongTensor(segment_pos), 
                torch.LongTensor(attn_mask), 
                torch.LongTensor(mlm_labels), 
                torch.LongTensor(nsp_label)
            )
                  


class MLMDataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=True):
        self.proc_id = proc_id
        self.shuffle = shuffle
        self.proc_num = proc_num
        self.batch_size = batch_size
        self.buffer_size = args.instances_buffer_size
        # We only need to read dataset once when buffer is big enough to load entire dataset.
        self.repeat_read_dataset = False
        self.f_read = open(dataset_path, "rb")
        self.start = 0
        self.end = 0
        self.data_index = 0
        self.buffer = []
        
    def _fill_buf(self):
        if len(self.buffer) > 0 and not self.repeat_read_dataset:
            if self.shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)
        else:
            self.buffer = []
            while True:
                try:
                    instances = pickle.load(self.f_read)
                except EOFError:
                    # Reach file end.
                    if not self.repeat_read_dataset:
                        # Buffer is big enough to load entire dataset.
                        break
                    # Buffer is not big enough, read dataset form start.
                    self.f_read.seek(0)
                    self.data_index = 0
                    instances = pickle.load(self.f_read)

                if self.data_index % self.proc_num == self.proc_id:
                    self.buffer.extend(instances)
                self.data_index += 1

                if len(self.buffer) > self.buffer_size:
                    self.repeat_read_dataset = True 
                    break

            if self.shuffle:
                random.shuffle(self.buffer)
            self.start = 0
            self.end = len(self.buffer)

    def _empty(self):
        return self.start + self.batch_size >= self.end

    def __del__(self):
        self.f_read.close()

    def __iter__(self):
        while True:
            if self._empty():
                self._fill_buf()
            if not self.buffer:
                print("Warning: worker %d data buffer is empty." % self.proc_id)

            instances = self.buffer[self.start : self.start+self.batch_size] 
            self.start += self.batch_size
                
            input_ids, mlm_labels, segment_pos = [], [], []
            attn_mask = []
            for ins in instances:
                input_ids.append(ins[0])
                mlm_labels.append([0 for _ in ins[0]])
                for index,replace_id in ins[1]:
                    mlm_labels[-1][index] = replace_id

                segment_pos.append([1 for _ in ins[0]])
                for i in range(ins[2][0]):
                    segment_pos[-1][i] = 0
                attn_mask.append([1-x for x in segment_pos[-1]])
            
            yield (
                torch.LongTensor(input_ids), 
                torch.LongTensor(segment_pos), 
                torch.LongTensor(attn_mask), 
                torch.LongTensor(mlm_labels)
            )


LOADERS = {
    "bert": BertDataLoader, 
    "sop": BertDataLoader,
    "mlm": MLMDataLoader, 
    "sbo": MLMDataLoader
}