import os
import random
from datetime import datetime

import evaluate
import numpy as np
import torch


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)
