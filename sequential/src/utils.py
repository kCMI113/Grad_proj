import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def get_config(path):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def recall_at_k(k, true, pred):
    true = true.data.cpu().numpy()
    pred = (pred[:k]).data.cpu().numpy()
    return len(np.intersect1d(true, pred)) / len(true)


def ndcg_at_k(k, true, pred):
    true = true.data.cpu().numpy()
    pred = (pred[:k]).data.cpu().numpy()

    log2i = np.log2(np.arange(2, k + 2))
    dcg = np.sum(np.isin(pred, true) / log2i)  # rel_i = 1
    idcg = np.sum((1 / log2i)[: min(len(true), k)])

    return dcg / idcg


def simple_recall_at_k(k, rank):
    if rank < k:
        return 1
    return 0


def simple_recall_at_k_batch(k, rank):
    return (rank < k).int()


def simple_ndcg_at_k(k, rank):
    if rank < k:
        return 1 / torch.log2(rank + 2)
    return 0


def simple_ndcg_at_k_batch(k, rank):
    return torch.where(rank < k, 1 / torch.log2(rank + 2), torch.tensor(0.0))


def mk_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


"""
from LRU Repo
"""

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import optim as optim


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks=[1, 5, 10, 20, 40]):
    metrics = {}
    labels = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics["R%d" % k] = (
            (
                hits.sum(1)
                / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
            )
            .mean()
            .cpu()
            .item()
        )

        metrics["M%d" % k] = (
            (hits / torch.arange(1, k + 1).unsqueeze(0).to(labels.device))
            .sum(1)
            .mean()
            .cpu()
            .item()
        )

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
            dcg.device
        )
        ndcg = (dcg / idcg).mean()
        metrics["N%d" % k] = ndcg.cpu().item()

    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }

    def sums(self, format_string="{}"):
        return {
            format_string.format(name): meter.sum for name, meter in self.meters.items()
        }

    def counts(self, format_string="{}"):
        return {
            format_string.format(name): meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
