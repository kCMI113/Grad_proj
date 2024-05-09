from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .common import MultiHeadAttention, PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"):
        super().__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)

    def forward(self, input_enc, mask):
        q,k,v = input_enc, input_enc, input_enc
        output_enc, attn_dist = self.attention(q,k,v, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist
    
class DecoderBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"):
        super().__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.cross_attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)

    def forward(self, input_enc, img_emb, mask):
        q,k,v = img_emb, img_emb, img_emb
        output_enc, attn_dist = self.attention(q,k,v, mask)
        
        _q,_k,_v = output_enc, input_enc, input_enc
        output_enc, attn_dist = self.cross_attention(_q,_k,_v, mask)
        
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_item: int,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        use_linear: bool = True,  # True if using linear layer at last
        device: str = "cpu",
        **kwargs
    ):
        super(BERT4Rec, self).__init__()

        self.hidden_size = hidden_size
        self.num_item = num_item
        self.pos_emb = pos_emb
        self.use_linear = use_linear
        self.device = device
        self.hidden_size = hidden_size

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        if self.pos_emb:
            self.positional_emb = nn.Embedding(max_len, hidden_size)
            
        encoderblocks = [EncoderBlock(num_attention_heads,
                                      hidden_size,
                                      dropout_prob,
                                      hidden_act)
                        for _ in range(num_encoder_layers)]
        
        decoderblocks = [DecoderBlock(num_attention_heads,
                                      hidden_size,
                                      dropout_prob,
                                      hidden_act)
                        for _ in range(num_decoder_layers)]
        
        self.encoder_blocks = nn.ModuleList(encoderblocks)
        self.decoder_blocks = nn.ModuleList(decoderblocks)

        if self.use_linear:
            self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, log_seqs, img_emb, **kwargs):
        seqs = self.item_emb(log_seqs).to(self.device)
        attn_mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device)

        if self.pos_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))

        seqs = self.emb_layernorm(self.dropout(seqs))

        for block in self.encoder_blocks:
            seqs, _ = block(seqs, attn_mask)
            
        for block in self.decoder_blocks:
            seqs, _ = block(seqs, img_emb, attn_mask)

        out = self.out(seqs) if self.use_linear else seqs
        return out
