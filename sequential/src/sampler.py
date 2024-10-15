import numpy as np


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
    def __init__(self, n_sample, n_users, sim_items, n_gen=6):
        self.n_sample = n_sample
        self.sim_items = sim_items.to(self)
        self.cache_list = [[-1 for _ in range(n_gen)] for _ in range(n_users)]

    def __call__(self, user_id, target, target_gen_idx, pos_lists):
        neg_samples = []
        for _ in range(self.n_sample):
            gen_idx = np.random.randint(6)
            while gen_idx == target_gen_idx:  # sampling gen_image index
                gen_idx = np.random.randint(6)

            if self.cache_list[user_id][gen_idx] == -1:  # not in cache
                self.cache_list[user_id][gen_idx] = np.setdiff1d(
                    self.sim_items[target, gen_idx, :],
                    pos_lists[user_id],
                    assume_unique=True,
                )[0]

            neg_samples.append(self.cache_list[user_id][gen_idx])

        return neg_samples
