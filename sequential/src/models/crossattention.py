from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .bert import BERT4Rec
from .common import MultiHeadAttention, PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_attention_heads, hidden_size, dropout_prob
        )
        self.pointwise_feedforward = PositionwiseFeedForward(
            hidden_size, dropout_prob, hidden_act
        )

    def forward(self, input_enc, mask):
        q, k, v = input_enc, input_enc, input_enc
        output_enc, attn_dist = self.attention(q, k, v, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class DecoderBlock(nn.Module):
    def __init__(
        self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"
    ):
        super().__init__()
        self.SA_img = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.SA_txt = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.CA_img = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.CA_txt = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.pointwise_feedforward = PositionwiseFeedForward(
            hidden_size, dropout_prob, hidden_act
        )

    def forward(self, enc_out, img_emb, txt_emb, mask):
        q, k, v = img_emb, img_emb, img_emb
        img_out, _ = self.SA_img(q, k, v, mask)

        _q, _k, _v = img_out, enc_out, enc_out
        CA_img_out, attn_dist = self.CA_img(_q, _k, _v, mask)

        q, k, v = txt_emb, txt_emb, txt_emb
        txt_out, _ = self.SA_txt(q, k, v, mask)

        _q, _k, _v = txt_out, CA_img_out, CA_img_out
        CA_txt_out, attn_dist = self.CA_txt(_q, _k, _v, mask)

        block_out = self.pointwise_feedforward(CA_txt_out)
        return block_out, attn_dist


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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_item = num_item
        self.num_decoder_layers = num_hidden_layers
        self.use_linear = use_linear

        decoderblocks = [
            DecoderBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
            for _ in range(self.num_decoder_layers)
        ]

        self.decoder_blocks = nn.ModuleList(decoderblocks)

        if self.use_linear:
            self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, seqs, img_emb, txt_emb, attn_mask, **kwargs):
        for block in self.decoder_blocks:
            block_out, _ = block(seqs, img_emb, txt_emb, attn_mask)

        layer_out = self.out(block_out) if self.use_linear else block_out
        return layer_out


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
            **kwargs
        )

        self.Decoder = Decoder(
            num_item=num_item,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            use_linear=True,
        )

    def forward(self, log_seqs, modal_emb, **kwargs):
        encoder_out, attn_mask = self.Encoder(log_seqs)

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
        self.img_emb_size = 512
        self.txt_emb_size = 512

        if self.img_emb_size != self.hidden_size:
            self.projection_img = nn.Linear(self.img_emb_size, self.hidden_size)
        if self.txt_emb_size != self.hidden_size:
            self.projection_txt = nn.Linear(self.txt_emb_size, self.hidden_size)

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        if self.pos_emb:
            self.positional_emb = nn.Embedding(max_len, hidden_size)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
                for _ in range(self.num_hidden_layers)
            ]
        )

        if self.use_linear:
            self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, log_seqs, img_emb, txt_emb, **kwargs):
        seqs = self.item_emb(log_seqs).to(self.device)
        attn_mask = (
            (log_seqs > 0)
            .unsqueeze(1)
            .repeat(1, log_seqs.shape[1], 1)
            .unsqueeze(1)
            .to(self.device)
        )

        if img_emb.shape[-1] != self.hidden_size:
            img_emb = self.projection_img(img_emb)
        if txt_emb.shape[-1] != self.hidden_size:
            txt_emb = self.projection_txt(txt_emb)

        img_emb.to(self.device)
        txt_emb.to(self.device)

        if self.pos_emb:
            positions = np.tile(
                np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]
            )
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))

        seqs = self.emb_layernorm(self.dropout(seqs))

        for block in self.decoder_blocks:
            layer_out, _ = block(seqs, img_emb, txt_emb, attn_mask)

        out = self.out(layer_out) if self.use_linear else layer_out
        return out
