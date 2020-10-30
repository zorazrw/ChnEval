#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Evaluation using Cosine Similarity
@author: ZhiruoWang
"""


import torch
import argparse
from predictor import SimPreds
from chneval.modeling import MODELS
from utils import load_batch
from chneval.tokenizer import CKETokenizer



def create_dataset(test_path, tokenizer, seq_length=128):
    # print("Start building dataset ...")
    dataset = []
    question = []
    with open(test_path, "r", encoding='utf-8') as fr:
        for i, line in enumerate(fr):
            if i % 4 == 3:   # base sentence
                base_text, base_pos = question[0].strip().split('\t')
                a_text, a_pos = question[1].strip().split('\t')
                b_text, b_pos = question[2].strip().split('\t')
                
                base_pos = int(base_pos)
                base_word = base_text[base_pos]
                
                a_pos = int(a_pos)
                a_word = a_text[a_pos]
                
                b_pos = int(b_pos)
                b_word = b_text[b_pos]
                if max(base_pos, a_pos, b_pos) > (seq_length - 1):
                    question = []
                    continue
                
                base_ins = tokenizer.tokenize_for_eval(
                    query=base_text, answer=base_word, position=base_pos, 
                    max_length=seq_length, do_padding=True
                )
                a_ins = tokenizer.tokenize_for_eval(
                    query=a_text, answer=a_word, position=a_pos, 
                    max_length=seq_length, do_padding=True
                )
                b_ins = tokenizer.tokenize_for_eval(
                    query=b_text, answer=b_word, position=b_pos, 
                    max_length=seq_length, do_padding=True
                )
                if base_ins and a_ins and b_ins:
                    dataset.extend([base_ins, a_ins, b_ins])
                question = []
            else:
                question.append(line)
    assert(len(dataset) % 3 == 0)
    # print("Successfully built dataset of length {}!".format(len(dataset)))
    return dataset






def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # vocabulary options
    parser.add_argument("--vocab_path", type=str, default="../bert-base-pytorch/vocab.txt", help="Path of the vocabulary file.")
    
    # model configuration options
    parser.add_argument("--hidden_size", type=int, default=768, help="Size of the hidden states.")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Size of the intermediate layer.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum length of table string.")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of the atttention heads.")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of the hidden layers.")
    parser.add_argument("--hidden_dropout_prob", type=int, default=0.1, help="Dropout probability for hidden layers.")
    parser.add_argument("--attention_dropout_prob", type=int, default=0.1, help="Dropout probability for attention.")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    
    # training options
    parser.add_argument("--question_num", type=int, default=8, help="Number of 3-sentence question for each batch.")
    parser.add_argument("--seq_length", type=int, default=256, help="Length of pre-processed sequences.")

    # io options    
    parser.add_argument("--data_path", type=str, default="../data/meaning-of-words/semantic.txt")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path of the pretrained model.")
    
    # optimizer options.
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")

    # fine-tune options.
    parser.add_argument("--device_id", type=int, default=0, help="Single GPU assignment.")
    parser.add_argument("--target", choices=["bert", "mlm", "sbo", "sop"], default="bert")
    
    args = parser.parse_args()
    
    
    tokenizer  = CKETokenizer(args.vocab_path)
    args.vocab_size = len(tokenizer.vocab)

    # Model initialization
    model = MODELS[args.target](args)
    predictor = SimPreds[args.target](args, model)
    device = predictor.set_device(do_report=True)
    
    dataset = create_dataset(args.data_path, tokenizer, args.seq_length)
    
    query_ids = torch.LongTensor([sample[0] for sample in dataset])
    token_type_ids = torch.LongTensor([sample[1] for sample in dataset])
    attn_mask = torch.LongTensor([sample[2] for sample in dataset])
    mlm_labels = torch.LongTensor([sample[3] for sample in dataset])
    
    instances_num = len(dataset)
    print("The number of evaluation questions: ", instances_num // 3)
    predictor.eval()
        
    correct_num = 0.
    for i, (query_batch,type_batch,attn_batch,label_batch) in enumerate(
        load_batch(
            args.question_num*3, query_ids, token_type_ids, attn_mask, mlm_labels
        )
    ):
        with torch.no_grad():
            correct = predictor(
                query_batch.to(device), 
                type_batch.to(device), 
                attn_batch.to(device), 
                label_batch.to(device)
            )
            correct_num += correct.item()
        
    print("{} right predictions out of {} instances".format(int(correct_num), instances_num // 3))
    acc = correct_num / (instances_num // 3)
    print("Correct rate: {:.4f}\n".format(acc))
    
    

if __name__ == "__main__":
    main()
