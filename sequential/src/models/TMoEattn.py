from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MultiHeadAttention, PositionwiseFeedForward
from .ARattn import CLIPCAModel, SelfAttention
from .MoEattn import MoEAttenBlock, MoEClipCA


class TMoEClipCA(MoEClipCA):
    def __init__(
        self,
        num_item: int,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        num_enc_layers: int = 2,
        num_enc_heads: int = 3,
        num_gen_layers: int = 2,
        num_gen_heads: int = 2,
        img_emb_size: int = 512,
        text_emb_size: int = 512,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        num_experts: int = 3,
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        use_linear: bool = True,  # True if using linear layer at last
        logit_scale_init_value: float = 2.6592,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            num_item,
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            num_enc_layers,
            num_enc_heads,
            img_emb_size,
            text_emb_size,
            hidden_act,
            num_experts,
            max_len,
            dropout_prob,
            pos_emb,
            use_linear,
            logit_scale_init_value,
            device,
        )

        if self.hidden_size != self.text_emb_size:
            self.enc_in_proj = nn.Linear(self.hidden_size, self.text_emb_size)
        if self.text_emb_size != self.gen_emb_size:
            self.gen_in_proj = nn.Linear(self.text_emb_size, self.gen_emb_size)
        if self.text_emb_size + self.gen_emb_size != self.hidden_size:
            self.fuser = nn.Linear(
                self.text_emb_size + self.gen_emb_size, self.hidden_size
            )

        self.item_enc = nn.ModuleList(
            [
                SelfAttention(
                    num_enc_heads, self.text_emb_size, dropout_prob, hidden_act
                )
                for _ in range(num_enc_layers)
            ]
        )

        self.generator = nn.ModuleList(
            [
                SelfAttention(
                    num_gen_heads, self.gen_emb_size, dropout_prob, hidden_act
                )
                for _ in range(num_gen_layers)
            ]
        )

    def forward(self, log_seqs, ori_emb, text_emb):
        seqs = self.item_emb(log_seqs).to(self.device)

        attn_mask = torch.tril(
            log_seqs.unsqueeze(1)
            .repeat(1, log_seqs.shape[1], 1)
            .unsqueeze(1)
            .to(self.device)
        )  # padding + autoReg

        if self.pos_emb:
            positions = np.tile(
                np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]
            )
            seqs += self.positional_emb(torch.tensor(positions).to(self.device))
        # seqs = self.emb_layernorm(self.dropout(seqs))

        # get prompt
        enc_in = seqs
        if self.hidden_size != self.text_emb_size:
            enc_in = self.enc_in_proj(enc_in)

        for enc in self.item_enc:
            enc_in = enc(enc_in, attn_mask)

        prompt_res = enc_in

        if self.text_emb_size != self.gen_emb_size:
            enc_in = self.gen_in_proj(enc_in)

        for enc in self.generator:
            enc_in = enc(enc_in, attn_mask)

        gen_emb = enc_in

        mm_info = torch.concat((prompt_res, gen_emb), dim=-1)
        if mm_info.shape[-1] != self.hidden_size:
            mm_info = self.fuser(mm_info)

        ## MoE CA
        ori_emb = ori_emb.to(self.device)
        text_emb = text_emb.to(self.device)

        concated_input = torch.cat((seqs, ori_emb, text_emb), dim=-1)
        if concated_input.shape[-1] != self.hidden_size:
            concated_input = self.modal_projection(concated_input)

        for block in self.blocks:
            mm_info = block(concated_input, mm_info, attn_mask)

        layer_out = (
            self.out(mm_info)
            if self.use_linear
            else torch.matmul(mm_info, self.item_emb.weight.T)
        )

        return gen_emb, prompt_res, layer_out
