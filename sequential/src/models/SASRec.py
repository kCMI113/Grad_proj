"""
https://github.com/pmixer/SASRec.pytorch
"""

import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, dropout_prob):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_prob)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(
        self,
        num_user,
        num_item,
        hidden_size: int = 256,  # hidden size for multi head attention
        num_attention_heads: int = 2,  # number of heads for multi head attention
        num_hidden_layers: int = 2,  # number of transformer layers
        max_len: int = 70,  # length of input sequence
        dropout_prob: float = 0.2,  # dropout probability
        pos_emb: bool = True,
        device: str = "cpu",
        **kwargs
    ):
        super(SASRec, self).__init__()

        self.user_num = num_user
        self.item_num = num_item
        self.dev = device
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.pos_emb = pos_emb
        self.device = device
        self.hidden_size = hidden_size

        # TODO: loss += l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, hidden_size, padding_idx=0
        )
        if self.pos_emb:
            self.pos_emb = torch.nn.Embedding(max_len + 1, hidden_size, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=dropout_prob)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_hidden_layers):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout_prob, batch_first=True
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_size, dropout_prob)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):  # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.tensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        if self.pos_emb:
            poss = (
                torch.arange(1, log_seqs.shape[1] + 1, device=self.dev)
                .unsqueeze(0)
                .repeat(log_seqs.shape[0], 1)
            )
            # poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
            # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
            poss *= log_seqs != 0
            seqs += self.pos_emb(torch.tensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )
        # key_padding_mask = (log_seqs == 0)

        # seqs : batch x seq_len x emb_dim
        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)

            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs  # residual
            # seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        logits = torch.matmul(
            log_feats, self.item_emb.weight.T
        )  # batch x seq_len x n_item+1

        # pos_embs = self.item_emb(torch.tensor(pos_seqs).to(self.dev))
        # neg_embs = self.item_emb(torch.tensor(neg_seqs).to(self.dev))

        # pos_logits = (log_feat/s * pos_embs).sum(dim=-1)
        # neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return logits

    def predict(self, log_seqs):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        # final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        logits = torch.matmul(
            log_feats, self.item_emb.weight.T
        )  # batch x seq_len x n_item+1
        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
