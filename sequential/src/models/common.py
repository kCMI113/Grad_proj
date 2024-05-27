import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_units, dropout_prob):
        super(ScaledDotProductAttention, self).__init__()
        self.head_units = head_units
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))
        # dim of output : batchSize x num_head x seqLen x head_units
        output = torch.matmul(attn_dist, V)
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_units = self.hidden_size // self.num_attention_heads

        # query, key, value, output
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.attention = ScaledDotProductAttention(self.head_units, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(self.hidden_size, 1e-6)

    def forward(self, q, k, v, mask):
        residual = q  # residual connection
        batch_size, seqlen = q.size(0), q.size(1)

        Q = self.W_Q(q).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        K = self.W_K(k).view(batch_size, seqlen, self.num_attention_heads, self.head_units)
        V = self.W_V(v).view(batch_size, seqlen, self.num_attention_heads, self.head_units)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob, hidden_act="gelu"):
        super(PositionwiseFeedForward, self).__init__()
        activates = {"gelu": nn.GELU(), "mish": nn.Mish(), "silu": nn.SiLU()}

        self.W_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(hidden_size, 1e-6)
        self.activate = activates[hidden_act]

    def forward(self, x):
        residual = x
        output = self.W_2(self.activate(self.dropout(self.W_1(x))))
        output = self.layerNorm(self.dropout(output) + residual)
        return output
