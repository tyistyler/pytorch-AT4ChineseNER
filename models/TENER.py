
import numpy as np
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder, AdaptedTransformerEncoder

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from torch.autograd import Function
class GRL(Function):
    @staticmethod
    def forward(self, x):
        return x

    @staticmethod
    def backward(self, grad_output):
        output = grad_output.neg()
        return output

class Out_Cls(nn.Module):
    def __init__(self, output_dim, cls_label=2, dropout=0.4):
        super().__init__()
        self.output_dim = output_dim
        self.cls_label = cls_label
        self.linear = nn.Linear(self.output_dim, self.cls_label)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.3, device=None):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        '''
            query   =    [batch_size, query_len, hid_dim]
            key     =    [batch_size, query_len, hid_dim]
            value   =    [batch_size, query_len, hid_dim]
            mask    =    [batch_size, 1, 1, query_len]
        '''

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # print(Q)
        # print(self.scale)
        # print(V)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)    # [batch_size, n_heads, query_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)    # [batch_size, n_heads, key_len, head_dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)    # [batch_size, n_heads, value_len, head_dim]

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale    # [batch_size, n_heads, query_len, key_len]

        if mask is not None:    # mask    =    [batch_size, 1, 1, query_len]
            energy = energy.masked_fill(mask==0, -1e10)
        attention = torch.softmax(energy, dim=-1)                   # [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.dropout(attention), V)                # [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0,2,1,3).contiguous()                         # [batch_size, query_len, n_heads, head_dim]
        x = x.view(batch_size, -1, self.hid_dim)                    # [batch_size, query_len, hid_dim]

        x = self.fc_o(x)                                            # [batch_size, query_len, hid_dim]

        return x, attention


class SAModel(nn.Module):
    def __init__(self, ner_tag_vocab, cws_tag_vocab, ner_embed, cws_embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  ner_bi_embed=None, cws_bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None,
                 use_knowledge=False,
                 use_zen=False,
                 gram2id=None, task_ner=None, task_cws=None,
                 device=None, adv_feature
                 ):
        """
        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        :param use_knowledge: 是否使用stanford corenlp的知识
        :param feature2count: 字典, {"gram2count": dict, "pos_tag2count": dict, "chunk_tag2count": dict, "dep_tag2count": dict},
        :param
        """
        super().__init__()
        self.use_knowledge = use_knowledge
        self.use_zen = use_zen
        self.gram2id = gram2id
        self.ner_embed = ner_embed
        self.cws_embed = cws_embed

        embed_size = self.ner_embed.embed_size
        self.ner_bi_embed = None
        if ner_bi_embed is not None:
            self.ner_bi_embed = ner_bi_embed
            self.cws_bi_embed = cws_bi_embed
            embed_size += self.ner_bi_embed.embed_size

        self.in_fc_ner = nn.Linear(embed_size, d_model)
        self.in_fc_cws =  nn.Linear(embed_size, d_model)

        print('d_model:',d_model)
        self.lstm_ner  = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.attn_ner = MultiHeadAttentionLayer(hid_dim=256, n_heads=1, dropout=0.3, device=device)
        
        self.lstm_cws = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.attn_shared = MultiHeadAttentionLayer(hid_dim=256, n_heads=1, dropout=0.3, device=device)
        
        self.lstm_shared = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.attn_cws = MultiHeadAttentionLayer(hid_dim=256, n_heads=1, dropout=0.3, device=device)

        # 梯度翻转
        self.Grl = GRL()

        self.task_ner = task_ner    # [16, 2]
        self.task_cws = task_cws    # [16, 2]
        self.cls_fn = nn.CrossEntropyLoss(reduction='mean')

        self.output_dim = 256

        self.out_fc_ner = nn.Linear(self.output_dim * 2, len(ner_tag_vocab))
        self.out_fc_cws = nn.Linear(self.output_dim * 2, len(cws_tag_vocab))
        self.out_cls    = Out_Cls(self.output_dim, 2)
        self.fc_dropout = nn.Dropout(fc_dropout)
        
        self.adv_dropout = nn.Dropout(adv_feature)

        trans_ner = allowed_transitions(ner_tag_vocab, include_start_end=True)
        self.crf_ner = ConditionalRandomField(len(ner_tag_vocab), include_start_end_trans=True,
                                              allowed_transitions=trans_ner)

        trans_cws = allowed_transitions(cws_tag_vocab, include_start_end=True)
        self.crf_cws = ConditionalRandomField(len(cws_tag_vocab), include_start_end_trans=True,
                                              allowed_transitions=trans_cws)
                                              
    def _forward(self, is_ner, ner_chars, cws_chars, ner_target, cws_target, ner_bigrams=None, cws_bigrams=None):
        # NER transformer + shared transformer
        if is_ner:
            ner_mask = ner_chars.ne(0)
            ner_hidden = self.ner_embed(ner_chars)
            if self.ner_bi_embed is not None:
                ner_bigrams = self.ner_bi_embed(ner_bigrams)
                ner_hidden = torch.cat([ner_hidden, ner_bigrams], dim=-1)
            ner_hidden = self.in_fc_ner(ner_hidden)
            self.ner_pool_dim = ner_hidden.size(1)
            # print(self.ner_pool_dim)

            # Bi-LSTM + self-attention
            ner_encoder_output, (hn1, cn1) = self.lstm_ner(ner_hidden)
            # print('ner_encoder_output',ner_encoder_output.size())
            # print('ner_mask',ner_mask.size())
            ner_encoder_output, _ = self.attn_ner(ner_encoder_output,ner_encoder_output,ner_encoder_output,ner_mask.unsqueeze(1).unsqueeze(1))
            ner_shared_output, (hn2, cn2) = self.lstm_shared(ner_hidden)  # [16, seq_len, 256]
            # print('ner_shared_output',ner_shared_output.size())
            ner_shared_output, _ = self.attn_shared(ner_shared_output,ner_shared_output,ner_shared_output,ner_mask.unsqueeze(1).unsqueeze(1))
            # print('attn_ner_shared_output',ner_shared_output.size())

            ner_encoder_output = torch.cat([ner_encoder_output, ner_shared_output], dim=-1)
            ner_encoder_output = self.out_fc_ner(ner_encoder_output)
            ner_logits = F.log_softmax(ner_encoder_output, dim=-1)

            # 池化操作
            # print('ner_shared_output',ner_shared_output.size())
            max_pool_output = F.max_pool2d(ner_shared_output, (self.ner_pool_dim, 1))   # [16, 1, 256]
            # print('max_pool_output',max_pool_output.size())
            max_pool_output = torch.squeeze(max_pool_output)                        # [16, 256]
            # print('max_pool_output',max_pool_output.size())
            '''
                还需要反转梯度，进行线性变化
            '''
            adv_feature = self.Grl.apply(max_pool_output)
            adv_feature = self.adv_dropout(adv_feature)
            
            adv_logits = self.out_cls(adv_feature)
            task_ner = self.task_ner
            if adv_logits.size() != self.task_ner.size():
                task_ner = self.task_ner[:adv_logits.size(0)]
            # print('adv_logits',adv_logits.size())
            # print('task_ner', task_ner.size())
            adv_loss = self.cls_fn(adv_logits, task_ner)

        # CWS transformer + shared transformer
        else:
            cws_mask = cws_chars.ne(0)
            cws_hidden = self.cws_embed(cws_chars)
            if self.ner_bi_embed is not None:
                cws_bigrams = self.cws_bi_embed(cws_bigrams)
                cws_hidden = torch.cat([cws_hidden, cws_bigrams], dim=-1)
            cws_hidden = self.in_fc_cws(cws_hidden)
            self.cws_pool_dim = cws_hidden.size(1)

            cws_encoder_output, (hn3, cn3) = self.lstm_cws(cws_hidden)
            cws_encoder_output, _ = self.attn_cws(cws_encoder_output,cws_encoder_output,cws_encoder_output,cws_mask.unsqueeze(1).unsqueeze(1))
            cws_shared_output, (hn2, cn2) = self.lstm_shared(cws_hidden)
            cws_shared_output, _ = self.attn_shared(cws_shared_output,cws_shared_output,cws_shared_output,cws_mask.unsqueeze(1).unsqueeze(1))

            cws_encoder_output = torch.cat([cws_encoder_output, cws_shared_output], dim=-1)
            cws_encoder_output = self.out_fc_cws(cws_encoder_output)
            cws_logits = F.log_softmax(cws_encoder_output, dim=-1)

            # 池化操作
            # print('ner_shared_output', cws_encoder_output.size())
            max_pool_output = F.max_pool2d(cws_shared_output, (self.cws_pool_dim, 1))  # [16, 1, 256]
            max_pool_output = torch.squeeze(max_pool_output)  # [16, 256]
            '''
                还需要反转梯度，进行线性变化
            '''
            adv_feature = self.Grl.apply(max_pool_output)
            adv_feature = self.adv_dropout(adv_feature)
            # print('ner_max_pool_output', max_pool_output.size())
            # print('adv_feature', adv_feature.size())
            adv_logits = self.out_cls(adv_feature)
            adv_loss = self.cls_fn(adv_logits, self.task_cws)

        # new add end----------------------------------------------------------------
        if ner_target is None:
            if is_ner:
                ner_paths, _ = self.crf_ner.viterbi_decode(ner_logits, ner_mask)
                return {'pred': ner_paths, 'cws_pred': None}
            else:
                cws_paths, _ = self.crf_cws.viterbi_decode(cws_logits, ner_mask)

                return {'pred': None, 'cws_pred':cws_paths}
        else:
            if is_ner:
                ner_loss = self.crf_ner(ner_logits, ner_target, ner_mask)
                return {'loss': ner_loss, 'cws_loss': None, 'adv_loss':adv_loss}
            else:
                cws_loss = self.crf_cws(cws_logits, cws_target, cws_mask)
                return {'loss': None, 'cws_loss':cws_loss, 'adv_loss':adv_loss}

    def forward(self, is_ner, ner_chars, cws_chars, ner_target, cws_target, ner_bigrams=None, cws_bigrams=None):
        return self._forward(is_ner, ner_chars, cws_chars, ner_target, cws_target, ner_bigrams, cws_bigrams)

    def predict(self, is_ner, ner_chars, cws_chars, ner_bigrams=None, cws_bigrams=None):
        return self._forward(is_ner, ner_chars, cws_chars, ner_target=None, cws_target=None, ner_bigrams=ner_bigrams, cws_bigrams=cws_bigrams)
