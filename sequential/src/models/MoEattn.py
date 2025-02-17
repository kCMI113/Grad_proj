from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MultiHeadAttention, PositionwiseFeedForward
from .ARattn import CLIPCAModel


class Router(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_expaerts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w = nn.Linear(self.hidden_size, self.num_expaerts)

    def forward(self, x):
        gate_scores = F.softmax(self.w(x), dim=-1)
        # capacity = int(self.capacity_factor * x.size(0))
        gate_scores, top_k_idx = gate_scores.topk(1, dim=-1)
        # mask = torch.zeros_like(gate_scores).scatter_(1, top_k_idx, 1)
        # masked_gate_scores = gate_scores * mask
        # denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        # gate_scores = (masked_gate_scores / denominators) * capacity

        return gate_scores, top_k_idx


class MoEAttenBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        dropout_prob,
        num_experts,
        hidden_act="gelu",
    ):
        super().__init__()
        self.CA = MultiHeadAttention(num_attention_heads, hidden_size, dropout_prob)
        self.experts = nn.ModuleList(
            [
                PositionwiseFeedForward(hidden_size, dropout_prob, hidden_act)
                for _ in range(num_experts)
            ]
        )
        self.gate = Router(hidden_size, num_experts)

        self.layerNorm = nn.LayerNorm(hidden_size, 1e-6)

    def forward(self, concated_input, gen_emb, mask):
        residual_gen = gen_emb

        concated_input, gen_emb = (
            self.layerNorm(concated_input),
            self.layerNorm(gen_emb),
        )

        ca_out, _ = self.CA(gen_emb, concated_input, concated_input, mask)
        ca_out += residual_gen  # residual
        ln_ca_out = self.layerNorm(ca_out)  # pre-ln

        gate_scores, top_k_idx = self.gate(ca_out)
        selected_expert_outputs = torch.stack(
            [expert(ln_ca_out) for expert in self.experts], dim=-2
        )
        selected_expert_outputs = torch.gather(
            selected_expert_outputs,
            dim=-2,
            index=top_k_idx.unsqueeze(-1).expand(
                -1, -1, -1, selected_expert_outputs.size(-1)
            ),
        )
        # moe_output = (
        #     torch.sum(gate_scores * selected_expert_outputs, dim=-1) + ln_img_ca_out
        # )  # skip-connection
        moe_output = selected_expert_outputs.squeeze(-2) + ln_ca_out  # skip-connection
        return moe_output


class MoEClipCA(CLIPCAModel):
    def __init__(
        self,
        num_item: int,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        num_enc_layers: int = 2,
        num_enc_heads: int = 3,
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
            hidden_act,
            max_len,
            dropout_prob,
            pos_emb,
            use_linear,
            logit_scale_init_value,
            device,
        )

        self.text_emb_size = text_emb_size
        self.modal_projection = nn.Linear(
            self.gen_emb_size + self.text_emb_size + self.hidden_size, self.hidden_size
        )
        self.blocks = nn.ModuleList(
            [
                MoEAttenBlock(
                    num_attention_heads,
                    hidden_size,
                    dropout_prob,
                    num_experts,
                    hidden_act,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )

        self.modal_gate = (
            Router(hidden_size=self.gen_emb_size + self.text_emb_size, num_experts=2)
            if modal_gate
            else None
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

        enc_in = seqs
        if self.hidden_size != self.ori_emb_size:
            enc_in = self.enc_in_proj(enc_in)

        # item encoder
        for enc in self.item_enc:
            enc_in = enc(enc_in, attn_mask)

        encoding_res = enc_in
        gen_emb = encoding_res

        if self.hidden_size != self.ori_emb_size:
            gen_emb = self.gen_emb_proj(gen_emb)

        # MoE CA
        ori_emb = ori_emb.to(self.device)
        text_emb = text_emb.to(self.device)

        if self.modal_gate is not None:
            gate_scores = self.modal_gate(
                torch.concat((ori_emb, text_emb), dim=-1)
            )  # (image, text)
            ori_emb *= gate_scores[:, :, 0].unsqueeze(-1)
            text_emb *= gate_scores[:, :, 1].unsqueeze(-1)

        concated_input = torch.cat((seqs, ori_emb, text_emb), dim=-1)

        if concated_input.shape[-1] != self.hidden_size:
            concated_input = self.modal_projection(concated_input)

        for block in self.blocks:
            gen_emb = block(concated_input, gen_emb, attn_mask)

        layer_out = (
            self.out(gen_emb)
            if self.use_linear
            else torch.matmul(gen_emb, self.item_emb.weight.T)
        )

        return encoding_res, layer_out
