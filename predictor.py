#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downstream Predictors for Cloze and Multi-Choice Prediction (MCP) probes

@author: Zhiruo Wang
"""

import torch
import torch.nn as nn
import chneval.modeling as md



# %% Cloze predictors
def get_score(indices, labels):
    assert(indices.size()[0] == labels.size()[0])
    scores = (indices == labels).long()
    scores = torch.sum(scores, dim=1)
    scores = torch.sum(scores, dim=0)
    return scores

    
class BertPredictor(md.Model):
    def __init__(self, model):
        super(md.Model, self).__init__()
        self.model = model.bert
        self.head  = model.cls
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels, topk=10):
        hidden_states, pooled_output = self.model(input_ids, token_type_ids, attn_mask)
        mlm_logits, _ = self.head(hidden_states, pooled_output)
        mlm_logits = self.softmax(mlm_logits)
        
        mlm_labels = mlm_labels.contiguous().view(-1)
        mlm_logits = mlm_logits.contiguous().view(mlm_labels.size()[0], -1)
        mlm_logits = mlm_logits[mlm_labels > 0, :]
        mlm_labels = mlm_labels[mlm_labels > 0]
        
        loss = self.loss_fct(mlm_logits, mlm_labels)
        values, indices = torch.topk(mlm_logits, topk) # values/indices: [masked_num x topk]
        correct = get_score(indices, mlm_labels.contiguous().view(-1,1)) # label: [masked_num x 1]
        return correct, indices, loss


class MlmPredictor(md.Model):
    def __init__(self, model):
        super(md.Model, self).__init__()
        self.model = model.bert
        self.head = model.cls
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels, topk=10):
        hidden_states, pooled_output = self.model(input_ids, token_type_ids, attn_mask)
        mlm_logits, _ = self.head(hidden_states, pooled_output)
        mlm_logits = self.softmax(mlm_logits)
        
        mlm_labels = mlm_labels.contiguous().view(-1)
        mlm_logits = mlm_logits.contiguous().view(mlm_labels.size()[0], -1)
        mlm_logits = mlm_logits[mlm_labels > 0, :]
        mlm_labels = mlm_labels[mlm_labels > 0]
        
        loss = self.loss_fct(mlm_logits, mlm_labels)
        _, indices = torch.topk(mlm_logits, topk) # values/indices: [masked_num x topk]
        correct = get_score(indices, mlm_labels.contiguous().view(-1,1)) # label: [masked_num x 1]
        return correct, indices, loss
    

class SboPredictor(md.Model):
    def __init__(self, model):
        super(md.Model, self).__init__()
        self.model = model.bert
        self.head = model.cls
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels, topk=10):
        hidden_states, pooled_output = self.model(input_ids, token_type_ids, attn_mask)
        mlm_logits, _ = self.head(hidden_states, pooled_output)
        mlm_logits = self.softmax(mlm_logits)
        
        mlm_labels = mlm_labels.contiguous().view(-1)
        mlm_logits = mlm_logits.contiguous().view(mlm_labels.size()[0], -1)
        mlm_logits = mlm_logits[mlm_labels > 0, :]
        mlm_labels = mlm_labels[mlm_labels > 0]
        
        loss = self.loss_fct(mlm_logits, mlm_labels)
        _, indices = torch.topk(mlm_logits, topk) # values/indices: [masked_num x topk]
        correct = get_score(indices, mlm_labels.contiguous().view(-1,1)) # label: [masked_num x 1]
        return correct, indices, loss


 # %% Multi-choice predictors   
class BertMultiChoice(md.Model):
    def __init__(self, args, model):
        super(md.Model, self).__init__()
        self.model = model.bert
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.hidden_size = args.hidden_size
        
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels):
        """
        Args:
            input_ids: [3*question_num, seq_length, hidden_size]
            token_type_ids/attn_mask: size same as above
            mlm_labels: [3*question_num, seq_length, hidden_size], each sequence with one positive label
        """
        hidden_states, _ = self.model(input_ids, token_type_ids, attn_mask)  # [3*question_num, seq_length, hidden_size]
        
        mlm_labels = mlm_labels.contiguous().view(-1)  # [3*question_num x seq_length, hidden_size]
        logits = hidden_states.contiguous().view(mlm_labels.size()[0], -1)  # [3*question_num x seq_length, hidden_size]
        logits = logits[mlm_labels > 0, :]   # [3*question_num, hidden_size]  
        logits = logits.contiguous().view(-1, 3, self.hidden_size)
        base, correct, disturb = logits.contiguous().transpose(0, 1)
        
        correct_sim = self.cos(base, correct)
        disturb_sim = self.cos(base, disturb)
        output = (correct_sim > disturb_sim).float()
        pred = torch.sum(output)
        return pred


class MlmMultiChoice(md.Model):
    def __init__(self, args, model):
        super(md.Model, self).__init__()
        self.model = model.bert
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.hidden_size = args.hidden_size
        
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels):
        hidden_states, _ = self.model(input_ids, token_type_ids, attn_mask)

        mlm_labels = mlm_labels.contiguous().view(-1)  # [3*question_num x seq_length, hidden_size]
        logits = hidden_states.contiguous().view(mlm_labels.size()[0], -1)  # [3*question_num x seq_length, hidden_size]
        logits = logits[mlm_labels > 0, :]   # [3*question_num, hidden_size]  
        logits = logits.contiguous().view(-1, 3, self.hidden_size)
        base, correct, disturb = logits.contiguous().transpose(0, 1)
        
        correct_sim = self.cos(base, correct)
        disturb_sim = self.cos(base, disturb)
        output = (correct_sim > disturb_sim).float()
        pred = torch.sum(output)
        return pred
    

# %% Aggregated classes
ClozePreds = {
    "bert": BertPredictor, 
    "mlm": MlmPredictor, 
    "sop": BertPredictor, 
    "sbo": SboPredictor
}

SimPreds = {
    "bert": BertMultiChoice, 
    "mlm": MlmMultiChoice, 
    "sop" : BertMultiChoice, 
    "sbo": MlmMultiChoice
}