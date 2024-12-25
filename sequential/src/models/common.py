import math
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_autocast(precision, device):
    if precision == "amp":
        return lambda: torch.amp.autocast(device_type=device)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.amp.autocast(dtype=torch.bfloat16, device_type=device)
    elif precision == "fp16":
        return lambda: torch.amp.autocast(dtype=torch.float16, device_type=device)
    elif precision == "fp32":
        return lambda: torch.amp.autocast(dtype=torch.float32, device_type=device)
    else:
        return suppress


def contrastive_loss(logits, ignore_idx: int = -100) -> torch.Tensor:
    labels = torch.arange(logits.shape[0]).to(logits.device)
    labels[torch.sum(logits, dim=0) == 0] = -100
    labels = labels.type(torch.long)
    logits[logits.isnan()] = 0

    return nn.functional.cross_entropy(logits, labels, ignore_index=ignore_idx)


def clip_loss(item_embs, img_embs, logit_scale, epsilon: float = 1e-8) -> torch.Tensor:
    img_embs = img_embs / (img_embs.norm(p=2, dim=-1, keepdim=True) + epsilon)
    item_embs = item_embs / item_embs.norm(p=2, dim=-1, keepdim=True)
    logit_scale = logit_scale.exp()
    total_loss = 0
    seq_len = img_embs.shape[1]

    for i in range(seq_len):
        item_emb = item_embs[:, i, :]
        img_emb = img_embs[:, i, :]
        logits_per_item = torch.matmul(item_emb, img_emb.t()) * logit_scale

        item_loss = contrastive_loss(logits_per_item)
        image_loss = contrastive_loss(logits_per_item.t())

        total_loss += ((item_loss + image_loss) / 2.0) / seq_len

    return total_loss


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


## ******** pre_LN로 수정됨!!!! 나중에 다른 모델에도 적용해줘야 함 ********
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

    def forward(self, q, k, v, mask):  # input LN 적용된 결과!
        batch_size, seqlen = q.size(0), q.size(1)

        Q = self.W_Q(q).view(
            batch_size, seqlen, self.num_attention_heads, self.head_units
        )
        K = self.W_K(k).view(
            batch_size, seqlen, self.num_attention_heads, self.head_units
        )
        V = self.W_V(v).view(
            batch_size, seqlen, self.num_attention_heads, self.head_units
        )

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        output = self.dropout(self.W_O(output))  # 호출한 곳에서 residual 더해줘야 함!
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
