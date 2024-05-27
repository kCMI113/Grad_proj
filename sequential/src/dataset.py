import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# Refactoring dataset


class GenDataset(Dataset):
    def __init__(
        self,
        user_seq,
        origin_img_emb: Optional[torch.Tensor],
        gen_img_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        mask_prob: float = 0.15,
        closest_origin: bool = False,
        **kwargs
    ) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.origin_img_emb = origin_img_emb
        self.gen_img_emb = gen_img_emb
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.closest_origin = closest_origin

        self.pad_index = 0
        self.mask_index = self.num_item + 1

    def get_gen_sample(self, item_num):
        sampling = np.random.randint(len(self.gen_img_emb[item_num]))
        return self.gen_img_emb[item_num][sampling]

    def get_gen_closest_origin(self, item_num):
        gen_embeddings = self.gen_img_emb[item_num]
        ori_embedding = self.origin_img_emb[item_num]
        closet_gen_arg = torch.argmax(torch.sum(ori_embedding * gen_embeddings, dim=-1)).item()
        return self.gen_img_emb[item_num][closet_gen_arg]

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = self.user_seq[index]
        tokens = []
        labels = []
        img_emb = []

        get_gen = self.get_gen_closest_origin if self.closest_origin else self.get_gen_sample

        for s in user:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.mask_index)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s + 1)
                labels.append(s + 1)
                img_emb.append(get_gen(s))
            else:
                tokens.append(s + 1)
                labels.append(self.pad_index)
                img_emb.append(self.origin_img_emb[s])

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding

        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)

        img_emb = img_emb[-self.max_len :]
        modal_emb = torch.stack(img_emb)
        modal_emb = zero_padding2d(modal_emb)

        return (
            tokens,
            modal_emb,
            labels,
        )


class TestGenDataset(GenDataset):
    def __init__(
        self,
        user_seq,
        origin_img_emb: Optional[torch.Tensor],
        gen_img_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        num_gen_img: int = 1,
        max_len: int = 30,
        closest_origin: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            origin_img_emb=origin_img_emb,
            gen_img_emb=gen_img_emb,
            num_user=num_user,
            num_item=num_item,
            num_gen_img=num_gen_img,
            max_len=max_len,
            closest_origin=closest_origin,
        )

    def __getitem__(self, index):
        get_gen = self.get_gen_sample if self.closest_origin else self.get_gen_closest_origin

        user = self.user_seq[index]
        tokens = torch.tensor(user, dtype=torch.long) + 1
        labels = [0 for _ in range(self.max_len)]
        img_emb = []

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.mask_index  # masking
        tokens = tokens[-self.max_len :]
        labels = torch.tensor(labels, dtype=torch.long)

        mask_len = self.max_len - len(tokens)
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))

        tokens = zero_padding1d(tokens)

        for i in range(len(user) - 1):
            img_emb.append(self.origin_img_emb[user[i]])

        img_emb.append(get_gen(user[-1]))

        img_emb = img_emb[-self.max_len :]
        modal_emb = torch.stack(img_emb)
        modal_emb = zero_padding2d(modal_emb)

        return index, tokens, modal_emb, labels


# Description base datasets


class DescriptionDataset(Dataset):
    def __init__(
        self,
        user_seq,
        text_emb: Optional[torch.Tensor],
        mean: float,
        std,
        num_user: int,
        num_item: int,
        max_len: int = 30,
        mask_prob: float = 0.15,
        **kwargs
    ) -> None:
        self.user_seq = user_seq
        self.text_emb = text_emb
        self.mean = mean
        self.std = std
        self.noise_size = text_emb.shape[-1]

        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.pad_index = 0
        self.mask_index = self.num_item + 1

    def get_noise(self):
        if torch.is_tensor(self.std):
            return torch.normal(mean=self.mean, std=self.std)
        else:
            return torch.normal(self.mean, self.std, size=(self.noise_size,))

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        user = self.user_seq[index]
        tokens = []
        labels = []
        embedding = []

        for s in user:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.mask_index)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s + 1)
                labels.append(s + 1)

                embedding.append(self.text_emb[s] + self.get_noise())
            else:
                tokens.append(s + 1)
                labels.append(self.pad_index)
                embedding.append(self.text_emb[s])  # s는 item idx ( -1 해야할지도?)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding

        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)

        embedding = embedding[-self.max_len :]
        modal_emb = torch.stack(embedding)
        modal_emb = zero_padding2d(modal_emb)

        return (
            tokens,
            modal_emb,
            labels,
        )


class TestDescriptionDataset(DescriptionDataset):
    def __init__(
        self,
        user_seq,
        text_emb: Optional[torch.Tensor],
        mean: float,
        std,
        num_user: int,
        num_item: int,
        max_len: int = 30,
        **kwargs
    ) -> None:
        super().__init__(
            user_seq=user_seq,
            text_emb=text_emb,
            mean=mean,
            std=std,
            num_user=num_user,
            num_item=num_item,
            max_len=max_len,
        )

    def __getitem__(self, index):
        user = self.user_seq[index]
        tokens = torch.tensor(user, dtype=torch.long) + 1
        labels = [0 for _ in range(self.max_len)]
        embedding = []

        labels[-1] = tokens[-1].item()  # target
        tokens[-1] = self.mask_index  # masking
        tokens = tokens[-self.max_len :]
        labels = torch.tensor(labels, dtype=torch.long)

        mask_len = self.max_len - len(tokens)
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))

        tokens = zero_padding1d(tokens)

        for i in user:
            embedding.append(self.text_emb[i])
        embedding[-1] += self.get_noise()

        embedding = embedding[-self.max_len :]
        modal_emb = torch.stack(embedding)
        modal_emb = zero_padding2d(modal_emb)

        return index, tokens, modal_emb, labels
