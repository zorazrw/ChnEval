#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions

@author: Zhiruo Wang
"""



def load_batch(batch_size, input_ids, token_type_ids, attn_mask, mlm_labels):
    num = (input_ids.size()[0] + batch_size - 1) // batch_size
    for k in range(num):
        start, end = k*batch_size, (k+1)*batch_size
        yield (
            input_ids[start: end], 
            token_type_ids[start: end], 
            attn_mask[start: end], 
            mlm_labels[start: end]
        )
