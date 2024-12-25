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
        ori_img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
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
        self.ori_img_emb = ori_img_emb
        self.text_emb = text_emb
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
        text_emb = []

        for s in user[:-2]:  # without valid, test, (1~)
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
                neg.extend(
                    self.hnsampler(s - 1, self.gen_img_idx[s - 1], self.user_seq[index])
                    if self.hnsampler is not None and self.hnsampler.n_sample > 0
                    else []
                )
                neg.extend(
                    self.negsampler(user)
                    if self.negsampler is not None and self.negsampler.n_sample > 0
                    else []
                )
                img_emb.append(self.gen_img_emb[s - 1][self.gen_img_idx[s - 1]])
                # img_emb.append(self.gen_img_emb[labels[-1] - 1])

            else:
                tokens.append(s)
                labels.append(self.pad_index)
                img_emb.append(self.ori_img_emb[s - 1])

            text_emb.append(self.text_emb[s - 1])
            negs.append(
                torch.tensor(neg)
                if neg
                else (
                    torch.zeros(self.n_negs, dtype=torch.int16)
                    if self.n_negs > 0
                    else []
                )
            )

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        img_emb = img_emb[-self.max_len :]
        text_emb = text_emb[-self.max_len :]
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
        text_emb = zero_padding2d(torch.stack(text_emb))
        negs = zero_padding2d(torch.stack(negs))

        return (tokens, labels, negs, img_emb, text_emb)


class ARDataset(Dataset):
    def __init__(
        self,
        user_seq,
        gen_img_idx: Optional[torch.Tensor],
        gen_img_emb: Optional[torch.Tensor],
        ori_img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        type: str = "Train",
        **kwargs
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.gen_img_idx = gen_img_idx
        self.gen_img_emb = gen_img_emb
        self.ori_img_emb = ori_img_emb
        self.text_emb = text_emb
        self.max_len = max_len
        self.type_idx = -3 if type == "Train" else (-2 if type == "Valid" else -1)

        self.pad_index = 0
        self.mask_index = self.num_item + 1

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = np.array(self.user_seq[index]) + 1  # item index range : (1,n_items)
        tokens = user[: self.type_idx]
        negs = []
        labels = user[1 : len(user) + self.type_idx + 1]
        gen_emb = []
        ori_emb = []
        # text_emb = []

        for i in range(len(tokens)):
            # gen_emb.append(
            # self.gen_img_emb[labels[i] - 1][self.gen_img_idx[labels[i] - 1]]
            # )  # target's gen img
            gen_emb.append(self.gen_img_emb[labels[i] - 1])  # target's gen img
            ori_emb.append(self.ori_img_emb[tokens[i] - 1])  # item ori img
            negs.append([])

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        gen_emb = gen_emb[-self.max_len :]
        ori_emb = ori_emb[-self.max_len :]
        negs = negs[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        if self.type_idx != -3:
            labels = [0 for _ in range(len(labels) - 1)] + [labels[-1]]

        # padding
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))  # padding left
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))  # padding top

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)

        gen_emb = zero_padding2d(torch.stack(gen_emb))
        ori_emb = zero_padding2d(torch.stack(ori_emb))
        # negs = zero_padding2d(torch.stack(negs))

        if self.type_idx == -3:
            return (tokens, labels, gen_emb, ori_emb)

        return (index, tokens, labels, gen_emb, ori_emb)


class CLIPCADataset(Dataset):
    def __init__(
        self,
        user_seq,
        ori_img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        type: str = "Train",
        **kwargs
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.ori_img_emb = ori_img_emb
        self.text_emb = text_emb
        self.max_len = max_len
        self.type_idx = -3 if type == "Train" else (-2 if type == "Valid" else -1)

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = np.array(self.user_seq[index]) + 1  # item index range : (1,n_items)
        tokens = user[: self.type_idx]
        labels = user[1 : len(user) + self.type_idx + 1]
        ori_emb = []
        gen_emb = []
        text_emb = []

        for i in range(len(tokens)):
            gen_emb.append(self.ori_img_emb[labels[i] - 1])  # target's gen img
            ori_emb.append(self.ori_img_emb[tokens[i] - 1])  # item ori img
            text_emb.append(self.text_emb[tokens[i] - 1])  # item text emb

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        ori_emb = ori_emb[-self.max_len :]
        gen_emb = gen_emb[-self.max_len :]
        text_emb = text_emb[-self.max_len :]

        mask_len = self.max_len - len(tokens)

        if self.type_idx != -3:
            labels = [0 for _ in range(len(labels) - 1)] + [labels[-1]]

        # padding
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))  # padding left
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))  # padding top

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)
        ori_emb = zero_padding2d(torch.stack(ori_emb))
        text_emb = zero_padding2d(torch.stack(text_emb))
        gen_emb = zero_padding2d(torch.stack(gen_emb))

        if self.type_idx == -3:
            return (tokens, labels, ori_emb, gen_emb, text_emb)

        return (index, tokens, labels, ori_emb, gen_emb, text_emb)


class TestDataset(Dataset):
    def __init__(
        self,
        user_seq,
        gen_img_idx: Optional[torch.Tensor],
        gen_img_emb: Optional[torch.Tensor],
        ori_img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        n_negs: int,
        max_len: int = 30,
        is_valid: bool = False,
        hnsampler=None,
        negsampler=None,
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.gen_img_idx = gen_img_idx
        self.gen_img_emb = gen_img_emb
        self.ori_img_emb = ori_img_emb
        self.text_emb = text_emb
        self.max_len = max_len
        self.is_valid = is_valid
        self.mask_index = self.num_item + 1
        self.n_negs = n_negs
        self.hnsampler = hnsampler
        self.negsampler = negsampler

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = np.array(self.user_seq[index]) + 1

        tokens = user[:-1] if self.is_valid else user[:]
        labels = [0 for _ in range(self.max_len)]
        img_emb = []
        text_emb = []
        negs = (
            [
                torch.tensor([0 for _ in range(self.n_negs)])
                for _ in range(self.max_len - 1)
            ]
            if self.is_valid
            and self.hnsampler is not None
            and self.negsampler is not None
            else []
        )

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.mask_index  # masking
        tokens = tokens[-self.max_len :]

        temp_neg = []

        if self.is_valid:
            temp_neg.extend(
                self.hnsampler(
                    labels[-1] - 1,
                    self.gen_img_idx[labels[-1] - 1],
                    self.user_seq[index],
                )
                if self.hnsampler is not None and self.hnsampler.n_sample > 0
                else []
            )
            temp_neg.extend(
                self.negsampler(user)
                if self.negsampler is not None and self.negsampler.n_sample > 0
                else []
            )
            negs.append(torch.tensor(temp_neg))

        labels = torch.tensor(labels, dtype=torch.long)
        tokens = torch.tensor(tokens, dtype=torch.long)

        for item in tokens[:-1]:  # 1~
            # img_emb.append(self.gen_img_emb[item - 1][self.gen_img_idx[item - 1]])
            img_emb.append(self.ori_img_emb[item - 1])
            text_emb.append(self.text_emb[item - 1])

        # # img_emb.append(
        # #     self.gen_img_emb[labels[-1] - 1][self.gen_img_idx[labels[-1] - 1]]
        # # )
        # img_emb.append(self.ori_img_emb[item - 1])
        img_emb.append(self.gen_img_emb[labels[-1] - 1])
        text_emb.append(self.text_emb[labels[-1] - 1])

        mask_len = self.max_len - len(tokens)
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))

        tokens = zero_padding1d(tokens)

        img_emb = img_emb[-self.max_len :]
        text_emb = text_emb[-self.max_len :]

        img_emb = zero_padding2d(torch.stack(img_emb))
        text_emb = zero_padding2d(torch.stack(text_emb))
        negs = torch.stack(negs) if len(negs) > 0 else torch.tensor([])

        return (index, tokens, labels, negs, img_emb, text_emb)
