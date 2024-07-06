from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .bert import BERT4Rec
from .common import MultiHeadAttention, PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"):
        super().__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)

    def forward(self, input_enc, mask):
        q, k, v = input_enc, input_enc, input_enc
        output_enc, attn_dist = self.attention(q, k, v, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class DecoderBlock(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"):
        super().__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.cross_attention = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)

    def forward(self, enc_out, img_emb, mask):
        q, k, v = img_emb, img_emb, img_emb
        img_out, _ = self.attention(q, k, v, mask[0])

        _q, _k, _v = img_out, enc_out, enc_out
        output_dec, attn_dist = self.cross_attention(_q, _k, _v, mask[1])

        output_enc = self.pointwise_feedforward(output_dec)
        return output_enc, attn_dist


class Decoder(nn.Module):
    def __init__(
        self,
        num_item: int,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        dropout_prob: float = 0.2,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        use_linear: bool = True,
        auto_regressive: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_item = num_item
        self.num_decoder_layers = num_hidden_layers
        self.use_linear = use_linear
        self.auto_regressive = auto_regressive

        decoderblocks = [
            DecoderBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
            for _ in range(self.num_decoder_layers)
        ]

        self.decoder_blocks = nn.ModuleList(decoderblocks)

        if self.use_linear:
            self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, seqs, modal_emb, attn_mask, **kwargs):
        for block in self.decoder_blocks:
            modal_emb, _ = block(seqs, modal_emb, attn_mask)

        out = self.out(modal_emb) if self.use_linear else modal_emb
        return out


class CA4Rec(nn.Module):
    def __init__(
        self,
        num_item: int,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        use_linear: bool = True,  # True if using linear layer at last
        device: str = "cpu",
        auto_regressive: bool = False,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_item = num_item
        self.pos_emb = pos_emb
        self.use_linear = use_linear
        self.device = device
        self.num_encoder_layers = num_hidden_layers
        self.num_decoder_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.img_embedding_size = 512
        self.auto_regressive = auto_regressive

        self.Encoder = BERT4Rec(
            num_item=num_item,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            max_len=max_len,
            dropout_prob=dropout_prob,
            pos_emb=pos_emb,
            use_linear=False,
            device=device,
            **kwargs,
        )

        self.Decoder = Decoder(
            num_item=num_item,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            use_linear=True,
            auto_regressive=auto_regressive,
        )

        if self.img_embedding_size != self.hidden_size:
            self.projection = nn.Linear(self.img_embedding_size, self.hidden_size)

    def forward(self, log_seqs, modal_emb, **kwargs):
        encoder_out, mask_pad = self.Encoder(log_seqs)

        if self.img_embedding_size != self.hidden_size:
            modal_emb = self.projection(modal_emb)
        modal_emb.to(self.device)

        if self.auto_regressive:
            mask_time = torch.tril(torch.ones(mask_pad.shape), diagonal=0).bool()
            self_attn_mask = (mask_pad & mask_time).to(self.device)
        else:
            self_attn_mask = mask_pad

        attn_mask = [self_attn_mask, mask_pad]
        out = self.Decoder(encoder_out, modal_emb, attn_mask)

        return out


class DOCA4Rec(nn.Module):
    def __init__(
        self,
        num_item: int,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        use_linear: bool = True,  # True if using linear layer at last
        device: str = "cpu",
        auto_regressive: bool = False,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_item = num_item
        self.pos_emb = pos_emb
        self.use_linear = use_linear
        self.device = device
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.img_embedding_size = 512
        self.auto_regressive = auto_regressive

        if self.img_embedding_size != self.hidden_size:
            self.projection = nn.Linear(self.img_embedding_size, self.hidden_size)

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        if self.pos_emb:
            self.positional_emb = nn.Embedding(max_len, hidden_size)

        decoderblocks = [
            DecoderBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
            for _ in range(self.num_hidden_layers)
        ]

        self.decoder_blocks = nn.ModuleList(decoderblocks)

        if self.use_linear:
            self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, log_seqs, modal_emb, **kwargs):
        seqs = self.item_emb(log_seqs).to(self.device)  # batch x seqlen x emb_size

        mask_pad = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device)
        if self.auto_regressive:
            mask_time = torch.tril(torch.ones(mask_pad.shape), diagonal=0).bool()
            self_attn_mask = (mask_pad & mask_time).to(self.device)
        else:
            self_attn_mask = mask_pad

        attn_mask = [self_attn_mask, mask_pad]

        if self.img_embedding_size != self.hidden_size:
            modal_emb = self.projection(modal_emb)
        modal_emb.to(self.device)

        if self.pos_emb:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))

        seqs = self.emb_layernorm(self.dropout(seqs))

        for block in self.decoder_blocks:
            modal_emb, _ = block(seqs, modal_emb, attn_mask)

        out = self.out(modal_emb) if self.use_linear else modal_emb
        return out
