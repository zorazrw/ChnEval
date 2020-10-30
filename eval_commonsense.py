#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate on ConceptNet World Knowledge

@author: ZhiruoWang
"""

import torch
import argparse
from predictor import ClozePreds
from chneval.modeling import MODELS
from utils import load_batch
from chneval.tokenizer import CKETokenizer



def create_dataset(test_path, tokenizer, seq_length):
    """Dataset with no natural sentence, defined by manual templates"""
    # print("Start building dataset ...")
    dataset = []
    with open(test_path, "r", encoding='utf-8') as fr:
        for line in fr:
            query, answer = line.strip().split('\t')
            instance = tokenizer.tokenize_for_eval(query=query, answer=answer)
            # instance = query_ids, token_type_ids, attn_mask, mlm_labels
            if instance:
                dataset.append(instance)
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
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the input batch.")
    parser.add_argument("--seq_length", type=int, default=128, help="Length of pre-processed sequences.")
    parser.add_argument("--topk",type=int, default=10,  help="Top k choice for cloze tasks.")

    # io options    
    parser.add_argument("--data_path", type=str, default="../data/commonsense/AtLocation.txt")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path of the pretrained model.")

    # fine-tune options.
    parser.add_argument("--device_id", type=int, default=0, help="Single GPU assignment.")
    parser.add_argument("--target", choices=["bert", "mlm", "sbo", "sop"], default="bert")
    
    args = parser.parse_args()
    
    
    tokenizer  = CKETokenizer(args.vocab_path)
    args.vocab_size = len(tokenizer.vocab)

    # Model initialization
    model = MODELS[args.target](args)    
    predictor = ClozePreds[args.target](model)
    device = predictor.set_device(do_report=True)

    dataset = create_dataset(args.data_path, tokenizer, seq_length=args.seq_length)
    query_ids = torch.LongTensor([sample[0] for sample in dataset])
    token_type_ids = torch.LongTensor([sample[1] for sample in dataset])
    attn_mask = torch.LongTensor([sample[2] for sample in dataset])
    mlm_labels = torch.LongTensor([sample[3] for sample in dataset])
    
    instances_num = len(dataset)
    print("The number of evaluation instances: ", instances_num)

    predictor.eval()                    
    corrects = 0.
    for i, (query_batch,type_batch,attn_batch,label_batch) in enumerate(
        load_batch(
            args.batch_size, query_ids, token_type_ids, attn_mask, mlm_labels
        )
    ):
        with torch.no_grad():
            correct, _, _ = predictor(
                query_batch.to(device), 
                type_batch.to(device), 
                attn_batch.to(device), 
                label_batch.to(device), 
                topk=args.topk
            )
            corrects += correct        
    print("{} right predictions out of {} instances".format(corrects, instances_num))
    acc = corrects / instances_num
    print("Correct rate: {:.4f}\n".format(acc))

    
    

if __name__ == "__main__":
    main()
