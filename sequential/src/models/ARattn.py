from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .common import MultiHeadAttention, PositionwiseFeedForward


class ARBlock(nn.Module):
    def __init__(
        self, num_attention_heads, hidden_size, dropout_prob, hidden_act="gelu"
    ):
        super().__init__()
        self.CA_img = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.CA_gen_item = MultiHeadAttention(
            num_attention_heads, hidden_size, dropout_prob
        )
        self.SA_item = MultiHeadAttention(
            num_attention_heads, hidden_size, dropout_prob
        )
        self.pointwise_feedforward = PositionwiseFeedForward(
            hidden_size, dropout_prob, hidden_act
        )
        self.layerNorm = nn.LayerNorm(hidden_size, 1e-6)

    def forward(self, item_emb, ori_emb, gen_emb, mask):
        residual_gen = gen_emb
        residual_item = item_emb

        item_emb, ori_emb, gen_emb = (
            self.layerNorm(item_emb),
            self.layerNorm(ori_emb),
            self.layerNorm(gen_emb),
        )

        img_ca_out, _ = self.CA_img(gen_emb, ori_emb, ori_emb, mask)
        img_ca_out += residual_gen  # residual

        # item_sa_out, _ = self.SA_item(item_emb, item_emb, item_emb)
        # item_sa_out += residual_item  # residual

        residual_gen_item = img_ca_out
        # img_ca_out, item_sa_out = self.layerNorm(img_ca_out), self.layerNorm(
        #     item_sa_out
        # )
        img_ca_out = self.layerNorm(img_ca_out)

        gen_item_ca_out, attn_dist = self.CA_gen_item(
            img_ca_out, item_emb, item_emb, mask
        )
        gen_item_ca_out += residual_gen_item  # residual

        block_out = self.pointwise_feedforward(self.layerNorm(gen_item_ca_out))
        return block_out, attn_dist


class ARModel(nn.Module):
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
        self.num_hidden_layers = num_hidden_layers
        self.use_linear = use_linear
        self.pos_emb = pos_emb
        self.device = device
        self.hidden_size = hidden_size
        self.gen_emb_size = 512
        self.ori_emb_size = 512

        if self.gen_emb_size != self.hidden_size:
            self.projection_gen = nn.Linear(self.gen_emb_size, self.hidden_size)
        if self.ori_emb_size != self.hidden_size:
            self.projection_ori = nn.Linear(self.ori_emb_size, self.hidden_size)

        self.item_emb = nn.Embedding(num_item + 2, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout_prob)
        self.emb_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        if self.pos_emb:
            self.positional_emb = nn.Embedding(max_len, hidden_size)

        self.blocks = nn.ModuleList(
            [
                ARBlock(num_attention_heads, hidden_size, dropout_prob, hidden_act)
                for _ in range(self.num_hidden_layers)
            ]
        )

        if self.use_linear:
            self.out = nn.Linear(self.hidden_size, self.num_item + 1)

    def forward(self, log_seqs, ori_emb, gen_emb):
        seqs = self.item_emb(log_seqs).to(self.device)
        attn_mask = torch.tril(
            log_seqs.unsqueeze(1)
            .repeat(1, log_seqs.shape[1], 1)
            .unsqueeze(1)
            .to(self.device)
        )

        if ori_emb.shape[-1] != self.hidden_size:
            ori_emb = self.projection_ori(ori_emb)
        if gen_emb.shape[-1] != self.hidden_size:
            gen_emb = self.projection_gen(gen_emb)

        ori_emb = ori_emb.to(self.device)
        gen_emb = gen_emb.to(self.device)

        if self.pos_emb:
            positions = np.tile(
                np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]
            )
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        for block in self.blocks:
            gen_emb, _ = block(seqs, ori_emb, gen_emb, attn_mask)

        layer_out = self.out(gen_emb) if self.use_linear else gen_emb
        return layer_out
