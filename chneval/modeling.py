#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modeling BERT, MLM, SBO, SOP

@author: Zhiruo Wang
"""

import math
import torch
import torch.nn as nn



def gelu(x):
    return (torch.erf(x/math.sqrt(2.0)) + 1.0) * x * 0.5


# %% Embedding
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        position_ids = torch.arange(input_ids.size()[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.size())
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)

        input_states = self.word_embeddings(input_ids)
        position_states = self.position_embeddings(position_ids)
        token_type_states = self.token_type_embeddings(token_type_ids)

        embeddings = input_states + position_states + token_type_states
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



# %% Encoder
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attn_mask):
        sequence_length = hidden_states.size()[1]
        attention_mask = (attn_mask>0).unsqueeze(1).repeat(1,sequence_length,1).unsqueeze(1)
        attention_mask = -10000.0 * (1.0 - attention_mask.float())

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs
        return hidden_states
    

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



# %% Objectives
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SBOPredictionHead(nn.Module):
    def __init__(self, config):
        super(SBOPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states, mlm_labels):
        hidden_states = self.transform(hidden_states)
        left_labels = torch.cat((mlm_labels[:,1:], mlm_labels[:,0].contiguous().view(-1,1)),-1)
        left_labels = left_labels.contiguous().view(-1)
        right_labels = torch.cat((mlm_labels[:,-1].contiguous().view(-1,1), mlm_labels[:,:-1]),-1)
        right_labels = right_labels.contiguous().view(-1)
        
        hidden_size = hidden_states.size()[-1]
        sbo_logits = hidden_states.contiguous().view(-1, hidden_size)
        sbo_logits = torch.cat((sbo_logits[left_labels>0,:],sbo_logits[right_labels>0,:]),-1)
        
        sbo_logits = self.decoder(sbo_logits)
        return sbo_logits


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score




# %% Model
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
    
    def init_weights(self, model_path):
        if model_path:
            self.load_state_dict(torch.load(model_path), strict=True)
            print("Parameters initiated from {}".format(model_path))
        else:
            for n,p in list(self.named_parameters()):
                if 'gamma' not in n and 'beta' not in n:
                    p.data.normal_(0, 0.02)
            print("Parameters initiated randomly")
   
    def init_weights_loose(self, model_path):
        for n,p in list(self.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
        print("Parameters first initiated randomly")   
        
        if model_path:
            pretrained_dict = torch.load(model_path)
            current_dict = self.state_dict()
            updated_dict = {k:v for (k,v) in pretrained_dict.items() if k in current_dict}
            self.load_state_dict(updated_dict, strict=False)
            print("{} parameters (pretrained: {}, current: {}) further initiated from {}".
                  format(len(updated_dict), len(pretrained_dict), len(current_dict), model_path))
    
    def set_device(self, do_report=False):
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))
            print("Model in 'cuda' Mode")
            if do_report:
                return torch.device("cuda")
        else:
            self.to(torch.device("cpu"))
            print("Model in 'cpu' Mode")
            if do_report:
                return torch.device("cpu")
    
    def save_model(self, save_path):
        if hasattr(self, "module"):
            torch.save(self.module.state_dict(), save_path)
        else:
            torch.save(self.state_dict(), save_path)




class PreTrain(Model):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.init_weights(config.pretrained_model_path)
        
    def masked_language_model(self, mlm_logits, mlm_labels):
        _, _, vocab_size = mlm_logits.size()
        mlm_labels = mlm_labels.contiguous().view(-1)
        mlm_logits = mlm_logits.contiguous().view(-1, vocab_size)
        mlm_logits = mlm_logits[mlm_labels > 0, :]
        mlm_labels = mlm_labels[mlm_labels > 0]
        mlm_loss = self.loss_fct(mlm_logits, mlm_labels)
        mlm_count = torch.tensor(mlm_logits.size(0) + 1e-6)
        mlm_correct = torch.sum((mlm_logits.argmax(dim=-1).eq(mlm_labels)).float())
        return mlm_loss, mlm_correct, mlm_count
    
    def next_sentence_prediction(self, nsp_logits, nsp_labels):
        nsp_loss = self.loss_fct(nsp_logits, nsp_labels)
        nsp_correct = self.softmax(nsp_logits).argmax(dim=-1).eq(nsp_labels).sum()
        return nsp_loss, nsp_correct





class BertModel(Model):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids, attn_mask):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attn_mask)
        pooled_output = self.pooler(encoder_outputs)
        return encoder_outputs, pooled_output



class BertForPreTraining(PreTrain):
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels, nsp_labels):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attn_mask)
        mlm_logits, nsp_logits = self.cls(sequence_output, pooled_output)
        mlm_loss, mlm_correct, mlm_count = self.masked_language_model(mlm_logits, mlm_labels)
        nsp_loss, nsp_correct = self.next_sentence_prediction(nsp_logits, nsp_labels)
        nsp_count = input_ids.size()[0]
        return (mlm_loss, mlm_correct, mlm_count), (nsp_loss, nsp_correct, nsp_count)



# mlm classes
class MlmModel(Model):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, token_type_ids, attn_mask):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attn_mask)
        return encoder_outputs 



class MlmForPreTraining(PreTrain):
    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels):
        sequence_output = self.bert(input_ids, token_type_ids, attn_mask)
        mlm_logits = self.mlm_head(sequence_output)
        mlm_loss, mlm_correct, mlm_count = self.masked_language_model(mlm_logits, mlm_labels)
        return mlm_loss, mlm_correct, mlm_count



# sbo class
class SboForPreTraining(PreTrain):
    def __init__(self, config):
        super(SboForPreTraining, self).__init__()
        self.bert = MlmModel(config)
        self.mlm_head = BertLMPredictionHead(config)
        self.sbo_head = SBOPredictionHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights_loose(config.pretrained_model_path)

    def subject_boundary_objective(self, sbo_logits, mlm_labels):
        mlm_labels = mlm_labels.contiguous().view(-1)
        mlm_labels = mlm_labels[mlm_labels > 0]
        
        sbo_loss = self.loss_fct(sbo_logits, mlm_labels)
        sbo_correct = torch.sum((sbo_logits.argmax(dim=-1).eq(mlm_labels)).float())
        sbo_count = torch.tensor(sbo_logits.size(0) + 1e-6)
        return sbo_loss, sbo_correct, sbo_count        

    def forward(self, input_ids, token_type_ids, attn_mask, mlm_labels):
        hidden_states = self.bert(input_ids, token_type_ids, attn_mask)
        
        mlm_logits = self.mlm_head(hidden_states)
        mlm_loss, mlm_correct, mlm_count = self.masked_language_model(mlm_logits, mlm_labels)
        
        sbo_logits = self.sbo_head(hidden_states, mlm_labels)
        sbo_loss, sbo_correct, sbo_count = self.subject_boundary_objective(sbo_logits, mlm_labels)
        return (mlm_loss, mlm_correct, mlm_count), (sbo_loss, sbo_correct, sbo_count)



MODELS = {
    "mlm": MlmForPreTraining,
    "sbo": SboForPreTraining, 
    "bert": BertForPreTraining, 
    "sop": BertForPreTraining
}
