# -*- encoding:utf-8 -*-
"""
Data Pre-process for BERT

@author: Zhiruo Wang
"""

import argparse
from data import DataZoo



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Path options.
    parser.add_argument("--input_path", type=str, required=True, 
        help="Input pre-training data.")
    parser.add_argument("--vocab_path", type=str, default="../data/vocab.txt", 
        help="Path of the vocabulary file.")
    parser.add_argument("--output_path", type=str, default="../data/dataset.pt", 
        help="Path to save the preprocessed dataset.")
    
    # Preprocess options.
    parser.add_argument("--seq_length", type=int, default=128, 
        help="Sequence length of instances.")
    parser.add_argument("--dup_factor", type=int, default=2, 
        help="Duplicate instances multiple times.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1, 
        help="Probability of truncating sequence.")
    parser.add_argument("--buffer_size", type=int, default=50000, 
        help="The buffer size of documents in memory.")
    parser.add_argument("--processes_num", type=int, default=1, 
        help="Split the whole dataset and process in parallel.")
    
    parser.add_argument("--target", choices=["bert", "mlm", "sbo", "sop"], 
        type=str, default="bert", help="Pre-training target.")
    
    args = parser.parse_args()
    
    # Build and save dataset.
    dataset = DataZoo[args.target](args)
    dataset.build_and_save(args.processes_num)


if __name__ == "__main__":
    main()
