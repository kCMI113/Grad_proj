import random
from random import sample
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(
        self,
        user_seq,
        sim_matrix,
        num_user,
        num_item,
        origin_img_emb,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        max_len: int = 30,
        mask_prob: float = 0.15,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_gen_img = num_gen_img
        self.origin_img_emb = origin_img_emb
        self.gen_img_emb = gen_img_emb
        self.idx_groups = idx_groups
        self.text_emb = text_emb
        self.img_noise = img_noise
        self.std = std
        self.mean = mean

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        tokens = []
        labels = []
        img_emb = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.num_item + 1)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s)
                labels.append(s)
                img_emb.append(self.gen_img_emb[s][np.random.randint(3)]) #s는 item idx ( -1 해야할지도?)
                negs.append(self.neg_sampler(s - 1, seq - 1) + 1)
            else:
                tokens.append(s)
                labels.append(0)
                img_emb.append(self.origin_img_emb[s]) #s는 item idx ( -1 해야할지도?)
                negs.append(np.zeros(self.neg_sample_size))

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        negs = negs[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        
        zero_padding = nn.ZeroPad1d((mask_len, 0))

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding(tokens)
        labels = zero_padding(tokens)
        
        modal_emb = torch.stack(img_emb, dtype=torch.float64)
        modal_emb = nn.ZeroPad2d((0,0,mask_len,0))(modal_emb)

        return (
            tokens,
            modal_emb,
            labels,
            negs,
        )


class BERTTestDataset(BERTDataset):
    def __init__(
        self,
        user_seq,
        sim_matrix,
        num_user,
        num_item,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        neg_sampling: bool = False,
        neg_size: int = 50,
        neg_sample_size: int = 3,
        max_len: int = 30,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            sim_matrix=sim_matrix,
            num_user=num_user,
            num_item=num_item,
            gen_img_emb=gen_img_emb,
            idx_groups=idx_groups,
            text_emb=text_emb,
            neg_sampling=neg_sampling,
            neg_size=neg_size,
            neg_sample_size=neg_sample_size,
            max_len=max_len,
            num_gen_img=num_gen_img,
            img_noise=img_noise,
            std=std,
            mean=mean,
        )

    def __getitem__(self, index):
        tokens = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        labels = [0 for _ in range(self.max_len)]
        negs = np.zeros((self.max_len, self.neg_sample_size))

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.num_item + 1  # masking
        negs[-1] = self.neg_sampler(labels[-1] - 1, tokens - 1) + 1

        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)
        tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)

        labels = torch.tensor(labels, dtype=torch.long)
        negs = torch.tensor(negs, dtype=torch.long)

        modal_emb = self.get_modal_emb(tokens, labels)

        return index, tokens, modal_emb, labels, negs
