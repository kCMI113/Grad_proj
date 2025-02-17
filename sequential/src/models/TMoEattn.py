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
        modal_gate: bool = False,
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
            modal_gate,
        )

        if self.hidden_size != self.text_emb_size:
            self.enc_in_proj = nn.Linear(self.hidden_size, self.text_emb_size)
        if self.text_emb_size != self.gen_emb_size:
            self.gen_in_proj = nn.Linear(self.text_emb_size, self.gen_emb_size)

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

        self.modal_projection = nn.Linear(
            self.gen_emb_size + self.hidden_size, self.hidden_size
        )

        if not modal_gate and (
            (self.text_emb_size + self.gen_emb_size) != self.hidden_size
        ):
            self.fuser = nn.Linear(
                self.text_emb_size + self.gen_emb_size,
                self.hidden_size,
            )
        if modal_gate and (self.text_emb_size != self.hidden_size):
            self.fuser = nn.Linear(
                self.text_emb_size,
                self.hidden_size,
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
        seqs = self.emb_layernorm(self.dropout(seqs))

        ## get prompt
        enc_in = seqs
        if self.hidden_size != self.text_emb_size:
            enc_in = self.enc_in_proj(enc_in)

        for enc in self.item_enc:
            enc_in = enc(enc_in, attn_mask)

        prompt_res = enc_in

        ## get gen_img
        if self.text_emb_size != self.gen_emb_size:
            enc_in = self.gen_in_proj(enc_in)

        for enc in self.generator:
            enc_in = enc(enc_in, attn_mask)

        gen_emb = enc_in

        ori_emb = ori_emb.to(self.device)
        text_emb = text_emb.to(self.device)

        ## Modal Gating
        if self.modal_gate is not None:
            gate_scores, top_k_idx = self.modal_gate(
                torch.concat((gen_emb, prompt_res), dim=-1)
            )  # (image, text)
            top_k_idx = top_k_idx.unsqueeze(-1)
            # batch X seg_len X 2 X emb_size
            gen_mm = torch.stack((gen_emb, prompt_res), dim=-2)
            gen_mm = torch.gather(
                gen_mm, dim=2, index=top_k_idx.expand(-1, -1, -1, gen_mm.size(-1))
            ).squeeze(-2)
            mm_info = torch.stack((ori_emb, text_emb), dim=-2)
            mm_info = torch.gather(
                mm_info, dim=2, index=top_k_idx.expand(-1, -1, -1, gen_mm.size(-1))
            ).squeeze(-2)
            concated_input = torch.cat((seqs, mm_info), dim=-1)

        else:
            concated_input = torch.cat((seqs, ori_emb, text_emb), dim=-1)
            mm_info = torch.concat((gen_emb, prompt_res), dim=-1)

        if concated_input.shape[-1] != self.hidden_size:
            concated_input = self.modal_projection(concated_input)

        if mm_info.shape[-1] != self.hidden_size:
            mm_info = self.fuser(mm_info)

        for block in self.blocks:
            mm_info = block(concated_input, mm_info, attn_mask)

        layer_out = (
            self.out(mm_info)
            if self.use_linear
            else torch.matmul(mm_info, self.item_emb.weight.T)
        )

        if self.modal_gate is not None:
            return (
                gen_emb,
                prompt_res,
                layer_out,
                torch.mean(top_k_idx.to(torch.float32)),
            )
        return gen_emb, prompt_res, layer_out, torch.tensor([-1])
