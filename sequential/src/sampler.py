import numpy as np
import torch
import torch.nn as nn


class NegSampler:
    def __init__(self, n_sample, n_users, n_items):
        self.n_sample = n_sample
        self.n_items = n_items
        self.all_items = list(range(1, self.n_items + 1))  # all_items
        self.cache_list = [None for _ in range(n_users)]

    def __call__(self, user_id, pos_lists):
        neg_samples = []
        if self.cache_list[user_id] is None:  # not in cache
            self.cache_list[user_id] = np.setdiff1d(
                self.all_items, pos_lists, assume_unique=True
            )

        for _ in range(self.n_sample):
            neg_samples.append(np.random.choice(self.cache_list[user_id]))
        return neg_samples


class HardNegSampler:
    def __init__(self, n_sample, gen_emb, ori_emb):
        self.n_sample = n_sample
        self.gen_emb = gen_emb.to("cpu")
        self.ori_emb = ori_emb.to("cpu")
        self.cos = nn.CosineSimilarity(dim=1)

    def __call__(self, target, target_gen_idx, pos_lists):
        neg_samples = []
        for _ in range(self.n_sample):
            gen_idx = np.random.randint(6)
            while gen_idx == target_gen_idx:  # sampling gen_image index
                gen_idx = np.random.randint(6)

            sim_items = torch.argsort(
                self.cos(self.ori_emb, self.gen_emb[target, gen_idx]), descending=True
            )[:1900]
            neg_samples.append(
                np.setdiff1d(
                    sim_items,
                    pos_lists,
                    assume_unique=True,
                )[0]
                + 1
            )

        return neg_samples
