import numpy as np
import torch
from src.model import BERT4Rec, BERT4RecWithHF, BPRLoss, MLPBERT4Rec, MLPRec
from src.utils import simple_ndcg_at_k_batch, simple_recall_at_k_batch
from torch import nn
from tqdm import tqdm


def train(model, optimizer, scheduler, dataloader, criterion, device):
    model.train()
    total_loss = 0

    for tokens, labels, negs in tqdm(dataloader):
        tokens = tokens.to(device)
        labels = labels.to(device)
        negs = negs.to(device)

        if isinstance(model, (MLPBERT4Rec, MLPRec)):
            logits = model(tokens, labels)
        if isinstance(model, (BERT4Rec, BERT4RecWithHF)):
            logits = model(tokens)

        if isinstance(criterion, (BPRLoss)):
            not_zero= torch.where(labels!=0)
            target_dist= logits[not_zero]
            pos_score = torch.gather(logits, -1, labels.unsqueeze(-1))
            neg_score = torch.gather(logits, -1, negs)                 #unsqueeze 안 써도됨.
            loss = criterion(pos_score, neg_score,target_dist, model.parameters())
        if isinstance(criterion, (nn.CrossEntropyLoss)):
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = criterion(logits, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # scheduler.step()
    return total_loss / len(dataloader)


def eval(
    model,
    mode,
    category_clue,
    num_gen_img,
    dataloader,
    criterion,
    train_data,
    item_prod_type,
    items_by_prod_type,
    device,
):
    model.eval()
    metrics_batches = {k: torch.tensor([]).to(device) for k in ["R10", "R20", "R40", "N10", "N20", "N40"]}
    total_loss = 0
    pred_list = []

    with torch.no_grad():
        for idx, (users, tokens, labels, negs) in enumerate(tqdm(dataloader)):
            tokens = tokens.to(device)
            labels = labels.to(device)
            users = users.to(device)
            negs = negs.to(device)

            if isinstance(model, (MLPBERT4Rec, MLPRec)):
                logits = model(tokens, labels)
            if isinstance(model, (BERT4Rec, BERT4RecWithHF)):
                logits = model(tokens)
            if mode == "valid":
                if isinstance(criterion, (BPRLoss)):
                    not_zero= torch.where(labels!=0)
                    target_dist= logits[not_zero]
                    pos_score = torch.gather(logits, -1, labels.unsqueeze(-1))
                    neg_score = torch.gather(logits, -1, negs)                 #unsqueeze 안 써도됨.
                    loss = criterion(pos_score, neg_score,target_dist, model.parameters())
                if isinstance(criterion, (nn.CrossEntropyLoss)):
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                total_loss += loss.item()

            used_items_batch = [np.unique(train_data[user]) for user in users.cpu().numpy()]

            target_batch = labels[:, -1]
            user_res_batch = -logits[:, -1, 1:]

            for i, used_item_list in enumerate(used_items_batch):
                user_res_batch[i][used_item_list] = user_res_batch[i].max() + 1

            same_cat_item_target = [
                items_by_prod_type[item_prod_type[target_single - 1].item()] for target_single in target_batch
            ]
            if category_clue and (not num_gen_img):
                mask_batch = torch.ones_like(user_res_batch, dtype=torch.bool).to(device)
                for i, mask_row in enumerate(same_cat_item_target):
                    # exclude same category items from index list
                    mask_batch[i][mask_row] = False
                    # remove other category items
                    user_res_batch[i][mask_batch[i]] = user_res_batch[i].max()

            # sorted item id e.g. [3452(1st), 7729(2nd), ... ]
            item_rank_batch = user_res_batch.argsort()

            if mode == "test":
                pred_list.append(torch.concat((item_rank_batch[:, :40], target_batch.unsqueeze(1)), dim=1))

            # rank of items e.g. index: item_id(0~), item_rank[0] : rank of item_id 0
            item_rank_batch = item_rank_batch.argsort()
            item_rank_batch = item_rank_batch.gather(dim=1, index=target_batch.view(-1, 1) - 1).squeeze()

            for k in [10, 20, 40]:
                recall = simple_recall_at_k_batch(k, item_rank_batch)
                ndcg = simple_ndcg_at_k_batch(k, item_rank_batch)

                metrics_batches["R" + str(k)] = torch.cat((metrics_batches["R" + str(k)], recall))
                metrics_batches["N" + str(k)] = torch.cat((metrics_batches["N" + str(k)], ndcg))

        for k in [10, 20, 40]:
            metrics_batches["R" + str(k)] = metrics_batches["R" + str(k)].mean()
            metrics_batches["N" + str(k)] = metrics_batches["N" + str(k)].mean()

        if mode == "valid":
            return total_loss / len(dataloader), metrics_batches

    pred_list = torch.cat(pred_list).tolist()
    return pred_list, metrics_batches
