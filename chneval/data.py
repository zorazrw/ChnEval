#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data set loaders for BERT and MLM styles

@author: Zhiruo Wang
"""


from __future__ import absolute_import
import os
import random
import pickle
from multiprocessing import Pool
from tokenizer import FullTokenizer



def count_lines(corpus_path):
    with open(corpus_path, "rb") as f:
        count = 0
        last_data = b'\n'
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1] != b'\n':
            count += 1
    return count



class BertDataset(object):
    def __init__(self, args):
        self.tokenizer = FullTokenizer(args.vocab_path)
        self.input_path = args.input_path
        self.output_path = args.output_path

        self.seq_length = args.seq_length
        self.dup_factor = args.dup_factor
        self.buffer_size = args.buffer_size        
        self.short_seq_prob = args.short_seq_prob

    def build_and_save(self, workers_num):
        lines_num = count_lines(self.input_path)
        print("Starting {} workers for building datasets ... ".format(workers_num))
        assert workers_num >= 1, "workers_num should be postive numbers."
        if workers_num == 1:
            self.worker(0, 0, lines_num)
            real_workers_num = 1
        else:
            assert lines_num >= workers_num * 10, "Small corpus, we recommend to reduce workers."
            pool = Pool(workers_num)
            real_workers_num = workers_num
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i+1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            
            if end < lines_num:
                pool.apply_async(func=self.worker, args=[i+1, end, lines_num])
                real_workers_num = workers_num + 1
            pool.close()
            pool.join()

        # Merge datasets.
        f_writer = open(self.output_path, "wb")
        for i in range(real_workers_num):
            tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
            while True:
                try:
                    instances = pickle.load(tmp_dataset_reader)
                    print("Ins: ", len(instances))
                    pickle.dump(instances, f_writer)
                except:
                    break
            tmp_dataset_reader.close()
            os.remove("dataset-tmp-"+str(i)+".pt")
        f_writer.close()

    def worker(self, process_id, start, end):
        print("Worker %d is building dataset ... " % process_id)
        all_instances = []
        docs_buffer = []
        document = []
        pos = 0
        f_write = open("dataset-tmp-" + str(process_id) + ".pt", "wb")
        with open(self.input_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    if not line:
                        break
                except:
                    continue
                finally:
                    pos += 1
                line = line.strip()
                if not line:
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.buffer_size:
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        all_instances.extend(instances)
                        print("Worker {:d}, process: {:.1f}%".format(process_id, (pos-start)/(end-start)*100))
                        # Clear buffer.
                        docs_buffer = []
                        instances = []
                    continue
                sentence = line.strip()
                if len(sentence) > 0:
                    document.append(sentence)
        
                if pos >= end - 1:
                    break

            if len(docs_buffer) > 0:
                instances = self.build_instances(docs_buffer)
                all_instances.extend(instances)
        random.shuffle(all_instances)
        pickle.dump(all_instances, f_write)
        f_write.close()

    def build_instances(self, all_documents):
        instances = []
        for doc_index in range(len(all_documents)):
            ins = self.create_bert(all_documents, doc_index)
            for _ in range(self.dup_factor):
                instances.extend(ins)
        print("Len Ins: ", len(instances))
        return instances

    def create_bert(self, all_documents, doc_index):
        document = all_documents[doc_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    text_a = ""
                    for j in range(a_end):
                        text_a += current_chunk[j]

                    text_b = ""
                    nsp_label = 0

                    # Random next
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        nsp_label = 1
                        target_b_length = target_seq_length - len(text_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != doc_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            text_b += random_document[j]
                            if len(text_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    # Actual next
                    else:
                        for j in range(a_end, len(current_chunk)):
                            text_b += current_chunk[j]
                    
                    input_ids, mlm_labels, segment_pos = self.tokenizer.tokenize_for_inputs(text_a=text_a, 
                                                                                            text_b=text_b, 
                                                                                            max_length=self.seq_length, 
                                                                                            do_padding=True)
                    instance = (input_ids, mlm_labels, nsp_label, segment_pos)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances




class MLMDataset(object):
    def __init__(self, args):
        self.tokenizer = FullTokenizer(args.vocab_path)
        self.input_path = args.input_path
        self.output_path = args.output_path

        self.seq_length = args.seq_length
        self.dup_factor = args.dup_factor
        self.buffer_size = args.buffer_size        
        self.short_seq_prob = args.short_seq_prob

    def build_and_save(self, workers_num):
        lines_num = count_lines(self.input_path)
        print("Starting {} workers for building datasets ... ".format(workers_num))
        assert workers_num >= 1, "workers_num should be postive numbers."
        if workers_num == 1:
            self.worker(0, 0, lines_num)
            real_workers_num = 1
        else:
            assert lines_num >= workers_num * 10, "Small corpus, we recommend to reduce workers."
            pool = Pool(workers_num)
            real_workers_num = workers_num
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i+1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            
            if end < lines_num:
                pool.apply_async(func=self.worker, args=[i+1, end, lines_num])
                real_workers_num = workers_num + 1
            pool.close()
            pool.join()

        # Merge datasets.
        f_writer = open(self.output_path, "wb")
        for i in range(real_workers_num):
            tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
            while True:
                try:
                     instances = pickle.load(tmp_dataset_reader)
                     pickle.dump(instances, f_writer)
                except:
                    break
            tmp_dataset_reader.close()
            os.remove("dataset-tmp-"+str(i)+".pt")
        f_writer.close()

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        sentences = []
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.input_path, mode="r", encoding="utf-8") as f:
            # locate to the starting position
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            
            # read needed lines
            while True:
                try:
                    line = f.readline()
                    if not line:  # reach file end
                        break
                except:
                    continue
                finally:
                    pos += 1
                
                sentence = line.strip()
                if len(sentence) > 0:
                    sentences.append(sentence)
                    
                if len(sentences) == self.buffer_size:
                    instances = self.build_instances(sentences)
                    pickle.dump(instances, f_write)
                    print("Worker {:d}, Process {:.1f}".format(proc_id, (pos-start)/(end-start)*100))
                    sentences = []  # clear buffer
                    # instances = []
                    
                if pos >= end - 1:  # reach region end
                    break
            if len(sentences) > 0:
                instances = self.build_instances(sentences)
                pickle.dump(instances, f_write)
        f_write.close()

    def build_instances(self, sentences):  # no dynamic masking
        instances = []        
        for sentence in sentences:
            for _ in range(self.dup_factor):
                instance = self.create_mlm(sentence)
                instances.append(instance)
        random.shuffle(instances)
        return instances

    def create_mlm(self, sentence):
        input_ids, mlm_labels, segment_pos = self.tokenizer.tokenize_for_inputs(
            text_a=sentence, 
            text_b=None, 
            max_length=self.seq_length, 
            do_padding=True
        )
        instance = (input_ids, mlm_labels, segment_pos)
        return instance




# SOPDataset
class SOPDataset(object):
    def __init__(self, args):
        self.tokenizer = FullTokenizer(args.vocab_path)
        self.input_path = args.input_path
        self.output_path = args.output_path

        self.seq_length = args.seq_length
        self.dup_factor = args.dup_factor
        self.buffer_size = args.buffer_size        
        self.short_seq_prob = args.short_seq_prob

    def build_and_save(self, workers_num):
        lines_num = count_lines(self.input_path)
        print("Starting {} workers for building datasets ... ".format(workers_num))
        assert workers_num >= 1, "workers_num should be postive numbers."
        if workers_num == 1:
            self.worker(0, 0, lines_num)
            real_workers_num = 1
        else:
            assert lines_num >= workers_num * 10, "Small corpus, we recommend to reduce workers."
            pool = Pool(workers_num)
            real_workers_num = workers_num
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i+1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            
            if end < lines_num:
                pool.apply_async(func=self.worker, args=[i+1, end, lines_num])
                real_workers_num = workers_num + 1
            pool.close()
            pool.join()

        # Merge datasets.
        f_writer = open(self.output_path, "wb")
        for i in range(real_workers_num):
            tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
            while True:
                try:
                     instances = pickle.load(tmp_dataset_reader)
                     pickle.dump(instances, f_writer)
                except:
                    break
            tmp_dataset_reader.close()
            os.remove("dataset-tmp-"+str(i)+".pt")
        f_writer.close()

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        docs_buffer = []
        document = []
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.input_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                    if not line:
                        break
                except:
                    continue
                finally:
                    pos += 1
                line = line.strip()
                if not line:
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.buffer_size:
                        # Build instances from documents.                    
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        pickle.dump(instances, f_write)
                        print("Worker {:d}, process: {:.1f}%".format(proc_id, (pos-start)/(end-start)*100))
                        # Clear buffer.
                        docs_buffer = []
                        instances = []
                    continue
                sentence = line.strip()
                if len(sentence) > 0:
                    document.append(sentence)
        
                if pos >= end - 1:
                    break

            if len(docs_buffer) > 0:
                instances = self.build_instances(docs_buffer)
                pickle.dump(instances, f_write)
        f_write.close()

    def build_instances(self, all_documents):
        instances = []
        for doc_index in range(len(all_documents)):
            ins = self.create_sop(all_documents, doc_index)
            for _ in range(self.dup_factor):
                instances.extend(ins)
        random.shuffle(instances)
        return instances

    def create_sop(self, all_documents, doc_index):
        document = all_documents[doc_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    text_a = ""
                    for j in range(a_end):
                        text_a += current_chunk[j]

                    text_b = ""                 
                    for j in range(a_end, len(current_chunk)):
                        text_b += current_chunk[j]
                    
                    sop_labels = 0
                    if random.random() < 0.5:
                        text_a, text_b = text_b, text_a
                        sop_labels = 1

                    input_ids, mlm_labels, segment_pos = self.tokenizer.tokenize_for_inputs(text_a=text_a, 
                                                                                            text_b=text_b, 
                                                                                            max_length=self.seq_length, 
                                                                                            do_padding=True)

                    instance = (input_ids, mlm_labels, sop_labels, segment_pos)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances



DataZoo = {
    "bert": BertDataset, 
    "mlm": MLMDataset,
    "sop": BertDataset, 
    "sbo": MLMDataset
}
