import numpy as np
import recbole.model.loss
import torch
import wandb
from src.models.ARattn import ARModel
from src.models.bert import BERT4Rec
from src.models.crossattention import (
    CA4Rec,
    DOCA4Rec,
    DOCAdvanced,
    MMDOCA4Rec,
    MMDOCAdvanced,
)
from src.models.mlp import MLPRec
from src.models.mlpbert import MLPBERT4Rec
from src.utils import simple_ndcg_at_k_batch, simple_recall_at_k_batch
from tqdm import tqdm


def train(accelerator, model, optimizer, scheduler, dataloader, criterion, device):
    model.train()
    total_loss = 0
    print("train str")
    with tqdm(dataloader) as t:
        # str_t = time.time()
        for tokens, labels, ori_emb, gen_emb in t:
            # with accelerator.accumulate(model):
            # str_loop = time.time()
            # print(str_loop - str_t)
            tokens = tokens.to(device)
            ori_emb = ori_emb.to(device)
            gen_emb = gen_emb.to(device)
            # negs = negs.to(device)
            labels = labels.to(device)

            # if isinstance(
            #     model,
            #     (MLPBERT4Rec, CA4Rec, DOCA4Rec, MMDOCA4Rec, DOCAdvanced, MMDOCAdvanced),
            # ):
            if isinstance(model, (ARModel)):
                logits = model(tokens, ori_emb, gen_emb)
            elif isinstance(model, BERT4Rec):
                logits, _ = model(log_seqs=tokens)
            # elif isinstance(model, MLPRec):
            #     logits = model(modal_emb)

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            if isinstance(criterion, recbole.model.loss.BPRLoss):
                neg_non_zero_idx = torch.where(negs[:, :, 0] != 0)
                pos_non_zero_idx = torch.where(labels != 0)
                neg_scores = torch.concat(
                    [
                        logits[
                            neg_non_zero_idx[0],
                            neg_non_zero_idx[1],
                            negs[neg_non_zero_idx[0], neg_non_zero_idx[1], i],
                        ]
                        for i in range(negs.shape[-1])
                    ]
                )
                pos_scores = (
                    logits[
                        pos_non_zero_idx[0],
                        pos_non_zero_idx[1],
                        labels[pos_non_zero_idx[0], pos_non_zero_idx[1]],
                    ]
                ).repeat(negs.shape[-1])
                loss = criterion(pos_scores, neg_scores)

            t.set_postfix(loss=loss.item())
            model.zero_grad()
            loss.backward()
            # accelerator.backward(loss)
            optimizer.step()
            wandb.log({"batch_loss": loss.item()})
            total_loss += loss.item()
            # print(time.time() - str_loop)

    # scheduler.step()
    return total_loss / len(dataloader)


def eval(
    model,
    mode,
    dataloader,
    criterion,
    train_data,
    device,
):
    model.eval()
    metrics_batches = {
        k: torch.tensor([]).to(device)
        for k in ["R1", "R5", "R10", "R20", "R40", "N1", "N5", "N10", "N20", "N40"]
    }
    total_loss = 0
    pred_list = []

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for users, tokens, labels, ori_emb, gen_emb in t:
                tokens = tokens.to(device)
                ori_emb = ori_emb.to(device)
                gen_emb = gen_emb.to(device)
                # negs = negs.to(device)
                labels = labels.to(device)

                if isinstance(model, (ARModel)):
                    logits = model(tokens, ori_emb, gen_emb)
                elif isinstance(model, BERT4Rec):
                    logits, _ = model(log_seqs=tokens)
                # elif isinstance(model, MLPRec):
                #     logits = model(modal_emb)

                if mode == "valid":
                    if isinstance(criterion, torch.nn.CrossEntropyLoss):
                        loss = criterion(
                            logits.view(-1, logits.size(-1)), labels.view(-1)
                        )
                    if isinstance(criterion, recbole.model.loss.BPRLoss):
                        neg_non_zero_idx = torch.where(negs[:, :, 0] != 0)
                        pos_non_zero_idx = torch.where(labels != 0)
                        neg_scores = torch.concat(
                            [
                                logits[
                                    neg_non_zero_idx[0],
                                    neg_non_zero_idx[1],
                                    negs[neg_non_zero_idx[0], neg_non_zero_idx[1], i],
                                ]
                                for i in range(negs.shape[-1])
                            ]
                        )
                        pos_scores = (
                            logits[
                                pos_non_zero_idx[0],
                                pos_non_zero_idx[1],
                                labels[pos_non_zero_idx[0], pos_non_zero_idx[1]],
                            ]
                        ).repeat(negs.shape[-1])
                        loss = criterion(pos_scores, neg_scores)
                    t.set_postfix(loss=loss.item())
                    wandb.log({"valid_batch_loss": loss.item()})
                    total_loss += loss

                used_items_batch = [
                    np.unique(train_data[user]) for user in users.cpu().numpy()
                ]
                target_batch = labels[:, -1]
                user_res_batch = -logits[:, -1, 1:]

                for i, used_item_list in enumerate(used_items_batch):
                    user_res_batch[i][used_item_list] = user_res_batch[i].max() + 1

                # sorted item id e.g. [3452(1st), 7729(2nd), ... ]
                item_rank_batch = user_res_batch.argsort()

                # if mode == "test":
                #     pred_list.append(
                #         torch.concat(
                #             (item_rank_batch[:, :40], target_batch.unsqueeze(1)), dim=1
                #         )
                #     )

                # rank of items e.g. index: item_id(0~), item_rank[0] : rank of item_id 0
                item_rank_batch = (
                    item_rank_batch.argsort()
                    .gather(dim=1, index=target_batch.view(-1, 1) - 1)
                    .squeeze()
                )

                for k in [1, 5, 10, 20, 40]:
                    recall = simple_recall_at_k_batch(k, item_rank_batch)
                    ndcg = simple_ndcg_at_k_batch(k, item_rank_batch)

                    metrics_batches["R" + str(k)] = torch.cat(
                        (metrics_batches["R" + str(k)], recall)
                    )
                    metrics_batches["N" + str(k)] = torch.cat(
                        (metrics_batches["N" + str(k)], ndcg)
                    )

        for k in [1, 5, 10, 20, 40]:
            metrics_batches["R" + str(k)] = metrics_batches["R" + str(k)].mean()
            metrics_batches["N" + str(k)] = metrics_batches["N" + str(k)].mean()

        if mode == "valid":
            return total_loss / len(dataloader), metrics_batches

    # pred_list = torch.cat(pred_list).tolist()
    return metrics_batches
