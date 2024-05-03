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
        user = self.user_seq[index]
        tokens = []
        labels = []
        img_emb = []

        for s in user:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.num_item + 1)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s+1)
                labels.append(s+1)
                img_emb.append(self.gen_img_emb[s][np.random.randint(3)]) #s는 item idx ( -1 해야할지도?)
            else:
                tokens.append(s+1)
                labels.append(0)
                img_emb.append(self.origin_img_emb[s]) #s는 item idx ( -1 해야할지도?)


        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0,0,mask_len,0))

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)
        
        img_emb = img_emb[-self.max_len :]
        modal_emb = torch.stack(img_emb)
        modal_emb.type(torch.float64)
        modal_emb = zero_padding2d(modal_emb)

        
        

        return (
            tokens,
            modal_emb,
            labels,
        )


class BERTTestDataset(BERTDataset):
    def __init__(
        self,
        user_seq,
        num_user,
        num_item,
        origin_img_emb,
        gen_img_emb: Optional[torch.Tensor] = None,
        idx_groups: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        max_len: int = 30,
        num_gen_img: int = 1,
        img_noise: bool = False,
        std: float = 1,
        mean: float = 0,
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            num_user=num_user,
            num_item=num_item,
            origin_img_emb=origin_img_emb,
            gen_img_emb=gen_img_emb,
            idx_groups=idx_groups,
            text_emb=text_emb,
            max_len=max_len,
            num_gen_img=num_gen_img,
            img_noise=img_noise,
            std=std,
            mean=mean,
        )

    def __getitem__(self, index):
        user = self.user_seq[index]
        
        tokens = torch.tensor(user, dtype=torch.long) + 1
        labels = [0 for _ in range(self.max_len)]
        img_emb = []
    

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.num_item + 1  # masking


        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0,0,mask_len,0))
        
        tokens = zero_padding1d(tokens)
        
        for i in range(len(user)):
            img_emb.append(self.origin_img_emb[user[i]])

        img_emb.append(self.gen_img_emb[user[-1]][np.random.randint(3)])
        img_emb = img_emb[-self.max_len:]
        
        modal_emb = torch.tensor(img_emb, dtype=torch.float64)
        modal_emb = zero_padding2d(modal_emb)
        

        return index, tokens, modal_emb, labels
