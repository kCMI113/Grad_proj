import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        user_seq,
        gen_img_idx: Optional[torch.Tensor],
        gen_img_emb: Optional[torch.Tensor],
        txt_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        n_negs: int,
        max_len: int = 30,
        mask_prob: float = 0.15,
        hnsampler=None,
        negsampler=None,
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.gen_img_idx = gen_img_idx
        self.gen_img_emb = gen_img_emb
        self.txt_emb = txt_emb
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.hnsampler = hnsampler
        self.negsampler = negsampler
        self.n_negs = n_negs

        self.pad_index = 0
        self.mask_index = self.num_item + 1

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = np.array(self.user_seq[index]) + 1  # item index range : (1,n_items)
        tokens = []
        negs = []
        labels = []
        img_emb = []
        txt_emb = []

        for s in user[:-2]:  # without valid, test
            prob = random.random()
            neg = []
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.mask_index)  # masking
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))  # random
                else:
                    tokens.append(s)  # origin
                labels.append(s)
                neg.extend(self.hnsampler(s - 1, self.gen_img_idx[s - 1], user))
                neg.extend(self.negsampler(index, user))
            else:
                tokens.append(s)
                labels.append(self.pad_index)

            img_emb.append(self.gen_img_emb[s - 1][self.gen_img_idx[s - 1]])
            txt_emb.append(self.txt_emb[s - 1])
            negs.append(
                torch.tensor(neg) if neg else torch.zeros(self.n_negs, dtype=torch.int8)
            )

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        img_emb = img_emb[-self.max_len :]
        txt_emb = txt_emb[-self.max_len :]
        negs = negs[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))  # padding left
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))  # padding top

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)

        img_emb = zero_padding2d(torch.stack(img_emb))
        txt_emb = zero_padding2d(torch.stack(txt_emb))
        negs = zero_padding2d(torch.stack(negs))

        return (tokens, labels, negs, img_emb, txt_emb)


class TestDataset(Dataset):
    def __init__(
        self,
        user_seq,
        gen_img_idx: Optional[torch.Tensor],
        gen_img_emb: Optional[torch.Tensor],
        txt_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        is_valid: bool = False,
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.gen_img_idx = gen_img_idx
        self.gen_img_emb = gen_img_emb
        self.txt_emb = txt_emb
        self.max_len = max_len
        self.is_valid = is_valid

    def __getitem__(self, index):
        end_idx = -2 if self.is_valid else -1
        user = np.ndarray(self.user_seq[index]) + 1

        tokens = user[:end_idx]
        labels = [0 for _ in range(self.max_len)]
        img_emb = []
        txt_emb = []

        labels[-1] = tokens[end_idx].item()  # target
        tokens[-1] = self.mask_index  # masking
        tokens = tokens[-self.max_len :]

        labels = torch.tensor(labels, dtype=torch.long)
        tokens = torch.tensor(tokens, dtype=torch.long)

        mask_len = self.max_len - len(tokens)
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))

        tokens = zero_padding1d(tokens)

        for item in range(user[:end_idx]):
            img_emb.append(self.gen_img_emb[item - 1][self.gen_img_idx[item - 1]])
            txt_emb.append(self.txt_emb[item - 1])

        img_emb = img_emb[-self.max_len :]
        txt_emb = txt_emb[-self.max_len :]

        img_emb = zero_padding2d(torch.stack(img_emb))
        txt_emb = zero_padding2d(torch.stack(txt_emb))

        return (index, tokens, labels, img_emb, txt_emb)
